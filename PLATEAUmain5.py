import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from scipy import stats

# === Step 0: Load data from Excel ===
df = pd.read_excel("FILE NAME.xlsx", sheet_name="SHEET NAME")

cols = ['Ref', 'Ads', 'Extr']
x_vars, y_vars = [], []

for col in cols:
    x = df[f'{col}X'].dropna().to_numpy()
    y = df[f'{col}Y'].dropna().to_numpy()
    x_vars.append(x)
    y_vars.append(y)

x1, x2, x3 = x_vars
y1, y2, y3 = y_vars

# Sort Line 1 and Line 2 by Y
sorted_idx1 = np.argsort(y1)
x1, y1 = x1[sorted_idx1], y1[sorted_idx1]

sorted_idx2 = np.argsort(y2)
x2, y2 = x2[sorted_idx2], y2[sorted_idx2]

# --- Step 1: Test if Line 1 intercept is significantly different from 0 ---
X1 = sm.add_constant(x1)
model1 = sm.OLS(y1, X1).fit()
intercept_pval = model1.pvalues[0]
slope_pval=model1.pvalues[1]
intercept = model1.params[0]
slope = model1.params[1]

# Plot
x_vals1 = np.linspace(min(x1), max(x1), 100)
x_vals2 = np.linspace(0, max(x1), 100)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(x1, y1, color='blue', label='Reference data')
ax1.set_xlabel(r'$TOC \/\ (ppm)$')
ax1.set_ylabel(r'$Dosage \/\ (mg/g)$')
ax1.grid(False)

# Popup
root = tk.Tk()
root.withdraw()

messagebox.showinfo("Step 1", "Testing if there is a meaningful relationship \nbetween signal and dosage..")
print("STEP 1: Testing slope of Line 1")
print(f"Intercept = {slope:.4f}, p-value = {slope_pval:.4f}")

if slope_pval < 0.05:
    messagebox.showinfo("Step 1", f"Slope = {slope:.4f}, p-value = {slope_pval:.4f}.\nSlope is significantly different from zero,\nthere is a meaningful realtionshipe between signal and dosage. Proceeding...\nPress OK")
    print("=> Slope is significantly different from zero. Proceeding...\n")

    messagebox.showinfo("Step 2", "Testing intercept of line 1...\nPress OK")
    print("STEP 2: Testing intercept of Line 1")
    print(f"Intercept = {intercept:.4f}, p-value = {intercept_pval:.4f}")

    if intercept_pval > 0.05:
        messagebox.showinfo("Step 2", f"Intercept = {intercept:.4f}, p-value = {intercept_pval:.4f}.\nIntercept is NOT significantly different from zero. Proceeding...\nPress OK")
        print("=> Intercept is NOT significantly different from zero. Proceeding...\n")

        # Initial combined arrays
        x_comb = np.concatenate([x1, x2])
        y_comb = np.concatenate([y1, y2])
        G = np.concatenate([np.zeros_like(x1), np.ones_like(x2)])
        interaction = x_comb * G
        X_interact = np.column_stack([x_comb, G, interaction])

        # Initial variances
        resid1_init = y1 - slope * x1
        var1_init = np.var(resid1_init, ddof=1)
        slope2_init = np.polyfit(x2, y2, 1)[0]
        resid2_init = y2 - slope2_init * x2
        var2_init = np.var(resid2_init, ddof=1)

        weights = np.concatenate([np.full_like(x1, 1 / var1_init), np.full_like(x2, 1 / var2_init)])

        # First WLS fit
        model_initial = sm.WLS(y_comb, X_interact, weights=weights).fit()

        # Updated variances
        fitted_vals = model_initial.fittedvalues
        residuals = y_comb - fitted_vals
        resid1_updated = residuals[G == 0]
        resid2_updated = residuals[G == 1]
        var1_updated = np.var(resid1_updated, ddof=1)
        var2_updated = np.var(resid2_updated, ddof=1)

        weights = np.concatenate([np.full_like(x1, 1 / var1_updated), np.full_like(x2, 1 / var2_updated)])

        # Refit WLS with updated weights
        model_final = sm.WLS(y_comb, X_interact, weights=weights).fit()
        slope_diff_pval = model_final.pvalues[2]

        # --- PARALLELISM LOOP ---
        while slope_diff_pval <= 0.05 and len(x2) > 3:

            # Remove lowest Y-value point
            removed_x, removed_y = x2[0], y2[0]
            x2, y2 = x2[1:], y2[1:]

            # Append to x3,y3 IN MEMORY (Option A)
            x3 = np.append(x3, removed_x)
            y3 = np.append(y3, removed_y)

            # Rebuild
            x_comb = np.concatenate([x1, x2])
            y_comb = np.concatenate([y1, y2])
            G = np.concatenate([np.zeros_like(x1), np.ones_like(x2)])
            interaction = x_comb * G
            X_interact = np.column_stack([x_comb, G, interaction])

            # Recompute variances
            resid1 = y1 - slope * x1
            var1 = np.var(resid1, ddof=1)
            slope2_tmp = np.polyfit(x2, y2, 1)[0]
            resid2 = y2 - slope2_tmp * x2
            var2 = np.var(resid2, ddof=1)

            weights = np.concatenate([np.full_like(x1, 1 / var1), np.full_like(x2, 1 / var2)])

            # Refit
            model_loop = sm.WLS(y_comb, X_interact, weights=weights).fit()
            slope_diff_pval = model_loop.pvalues[2]
            print(f"Retry parallelism test: slope diff p = {slope_diff_pval:.4f}, remaining points in Line2 = {len(x2)}")
        L=len(y3)
        # === At this point, parallelism holds OR we failed ===
        if slope_diff_pval > 0.05 and len(x2) >= 3:

            # ------------------------------------------------------------------
            #             VARIANCE MINIMIZATION LOOP (ALWAYS RUN)
            # ------------------------------------------------------------------

            def compute_variance(x2_test, y2_test, shared_slope):
                """Compute variance of residuals around Line 2: y = slope*x + intercept2."""
                Xtmp = sm.add_constant(x2_test)
                model_tmp = sm.OLS(y2_test - shared_slope * x2_test, np.ones_like(x2_test)).fit()
                intercept_tmp = model_tmp.params[0]
                residuals_tmp = y2_test - (shared_slope * x2_test + intercept_tmp)
                return np.var(residuals_tmp, ddof=1), intercept_tmp

            # Compute initial shared slope first
            X_parallel_initial = np.column_stack([x_comb, G])
            model_parallel_initial = sm.WLS(y_comb, X_parallel_initial, weights=weights).fit()
            shared_slope = model_parallel_initial.params[0]

            # Compute initial variance
            current_var, current_intercept2 = compute_variance(x2, y2, shared_slope)
            improvement = True

            while improvement and len(x2) > 3:

                # Try removing next lowest-y point
                test_x2 = x2[1:]
                test_y2 = y2[1:]

                test_var, _ = compute_variance(test_x2, test_y2, shared_slope)
                reduction = (current_var - test_var) / current_var

                if reduction >= 0.01:
                    # Accept removal
                    removed_x, removed_y = x2[0], y2[0]
                    x2, y2 = test_x2, test_y2
                    x3 = np.append(x3, removed_x)
                    y3 = np.append(y3, removed_y)
                    current_var = test_var
                    print(f"Variance reduced by {reduction*100:.2f}%, removing point ({removed_x}, {removed_y})")
                else:
                    improvement = False

            # AFTER variance minimization, rebuild arrays
            x_comb = np.concatenate([x1, x2])
            y_comb = np.concatenate([y1, y2])
            G = np.concatenate([np.zeros_like(x1), np.ones_like(x2)])

            # Final weights based on final dataset
            resid1 = y1 - slope * x1
            var1 = np.var(resid1, ddof=1)
            slope2_last = np.polyfit(x2, y2, 1)[0]
            resid2 = y2 - slope2_last * x2
            var2 = np.var(resid2, ddof=1)
            weights = np.concatenate([np.full_like(x1, 1 / var1), np.full_like(x2, 1 / var2)])

            # --- Final Step 3: shared slope model ---
            X_parallel = np.column_stack([x_comb, G])
            model_parallel = sm.WLS(y_comb, X_parallel, weights=weights).fit()

            slope = model_parallel.params[0]
            intercept2 = model_parallel.params[1]
            slope_pval = model_parallel.pvalues[0]
            intercept2_pval = model_parallel.pvalues[1]
            intercept2_se = model_parallel.bse[1]
            F=len(y3)-L

            #calculates CI, PI

            alpha = 0.05  # 95% interval
            df_resid = model_parallel.df_resid 

            resid_comb = y_comb - model_parallel.predict(X_parallel)
            sigma2 = np.var(resid_comb, ddof=1)

            var_intercept = model_parallel.cov_params()[1,1]

            se_pred = np.sqrt(sigma2 + var_intercept)

            t_val = stats.t.ppf(1 - alpha/2, df_resid)

            PI_lower = intercept2 - t_val * se_pred
            PI_upper = intercept2 + t_val * se_pred

            n2 = len(x2)  
            CI_lower = intercept2 - t_val * se_pred / np.sqrt(n2)
            CI_upper = intercept2 + t_val * se_pred / np.sqrt(n2)

            print(f"Prediction interval (single future point) for intercept: {PI_lower:.4f} to {PI_upper:.4f}")
            print(f"Confidence interval (true mean) for intercept: {CI_lower:.4f} to {CI_upper:.4f}")

            messagebox.showinfo(
                "Step 3",
                f"From the adsorption data:\n"
                f"{L} Points removed using prallelism possibility check, and {F} points removed using minimization of the variance critera.\n"
            )    
            messagebox.showinfo("Step 4", f"Constrained model with shared slope and Line 2 intercept:\nShared lope β = {slope:.4e}, p = {slope_pval:.4e}.\nLine 2 intercept α₂ = {intercept2:.4f}, SE = {intercept2_se:.4f}, p = {intercept2_pval:.4e}.\nPrediction interval for intercept: {PI_lower:.4f} to {PI_upper:.4f}.\nConfidence interval for intercept: {CI_lower:.4f} to {CI_upper:.4f}.\nPress OK")
            print("STEP 4: Constrained model with shared slope and Line 2 intercept:")
            print(f"Shared slope β = {slope:.4e}, p = {slope_pval:.4e}")
            print(f"Line 2 intercept α₂ = {intercept2:.4f}, SE = {intercept2_se:.4f}, p = {intercept2_pval:.4e}")
            

            if intercept2_pval<0.05:
                messagebox.showinfo("Step 4", f"All statistical tests passed successfully!\nYour plot will be shown.\nPress OK")        
                # === PLOT ===
                y_fit1 = slope * x_vals1
                y_fit2 = slope* x_vals2 + intercept2

                # Primary Y-axis plots
                ax1.plot(x_vals1, y_fit1, 'b--', label=f'Line 1: y = {slope:.3e}x')
                ax1.scatter(x2, y2, color='red', label='Adsorption data')
                ax1.scatter(x3, y3, facecolors='none', edgecolors='r', label='Adsorption data before saturation')
                ax1.plot(x_vals2, y_fit2, 'r--', label=f'Line 2: y = {slope:.3e}x + {intercept2:.4f}')
                #ax2.ax2.ax2
                            
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                ax1.legend(lines_1 , labels_1 , loc='best')



            else:
                messagebox.showerror("Step 4", "=> the intercept of the adsorption line is not significantly different from zero.\nEnd of calculations. Your plot will be shown.\nPress OK")
                print("=> the intercept of the adsorption line is not significantly different from zero")
                #intercept2=0

                y_fit1 = slope * x_vals1
                y_fit2 = slope* x_vals2 + intercept2
                ax1.plot(x_vals1, y_fit1, 'b--', label=f'Line 1: y = {slope:.3e}x')
                ax1.scatter(x2, y2, color='red', label='Adsorption data')
                ax1.scatter(x3, y3, facecolors='none', edgecolors='r', label='Adsorption data before saturation')
                ax1.plot(x_vals2, y_fit2, 'r--', label=f'Line 2: y = {slope:.3e}x + {intercept2:.4f}')

                # Legend
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                ax1.legend(lines_1, labels_1, loc='best')

        else:
            messagebox.showerror("Step 3", f"Even after exclusions, slopes are different or <3 points left.\nThe plot will show.\nPress OK")
            print("=> Cannot prove parallelism, stopping.")
            slope1, intercept1 = np.polyfit(x1, y1, 1)
            slope2, intercept2 = np.polyfit(x2, y2, 1)
            y_fit1 = slope1 * x_vals1 + intercept1
            y_fit2 = slope2 * x_vals2 + intercept2
            ax1.plot(x_vals1, y_fit1, 'b--', label=f'Line 1: y = {slope1:.3e}x+ {intercept1:.4f}')
            ax1.scatter(x2, y2, color='red', label='Adsorption data')
            ax1.scatter(x3, y3, facecolors='none', edgecolors='r', label='Adsorption data before saturation')
            ax1.plot(x_vals2, y_fit2, 'r--', label=f'Line 2: y = {slope2:.3e}x+ {intercept2:.4f}')
            ax1.legend(loc='best')

    else:
        messagebox.showerror("Step 2", f"Intercept = {intercept:.4f}, p-value = {intercept_pval:.4f}.\nIntercept IS significantly different from zero.\nCannot force Line 1 through origin.\nThe plot will show.\nPress OK")
        print("=> Intercept IS significantly different from zero. Cannot force Line 1 through origin.")
        slope, intercept = np.polyfit(x1, y1, 1)
        y_fit1 = slope * x_vals1 + intercept
        ax1.plot(x_vals1, y_fit1, 'b--', label=f'Line 1: y = {slope:.3e}x+ {intercept:.4f}')
        ax1.legend(loc='best')
    
else:
    messagebox.showerror("Step 1", f"Slope = {slope:.4f}, p-value = {slope_pval:.4f}.\nRef slope is NOT significantly different from zero.\nThere is no meaningful realtionship between signal and dosage.\nThe plot will show.\nPress OK")
    print("=> Ref slope is NOT significantly different from zero. There is no meaningful realtionship between signal and dosage.")
    slope, intercept = np.polyfit(x1, y1, 1)
    y_fit1 = slope * x_vals1 + intercept
    ax1.plot(x_vals1, y_fit1, 'b--', label=f'Line 1: y = {slope:.3e}x+ {intercept:.4f}')
    ax1.legend(loc='best')


plt.show()
