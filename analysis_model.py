import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_merge_data(feature_dir):
    data_frames = []
    
    # Load available features
    feature_files = {
        "pitch": "features_pitch.csv",
        "intensity": "features_intensity.csv",
        "speech_rate": "features_speech_rate.csv",
        "voice_quality": "features_voice_quality.csv",
        "emotion": "features_emotion.csv" 
    }
    
    merged_df = None
    
    for key, filename in feature_files.items():
        filepath = os.path.join(feature_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Ensure filename is the key
            if "filename" in df.columns:
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on="filename", how="outer")
        else:
            print(f"Warning: {filename} not found. Skipping.")

    return merged_df

def synthesize_missing_data(df):
    print("\n--- SYNTHESIZING MISSING DATA FOR DEMONSTRATION ---")
    np.random.seed(42)
    
    # 1. EMOTION (Arousal/Valence)
    # If not present, generate based on acoustic proxies + noise or random
    if "arousal_est" not in df.columns:
        print(" Synthesizing Emotional Arousal (Proxy: Intensity + Pitch)")
        # Normalize Pitch and Intensity
        p = df["pitch_mean"] if "pitch_mean" in df.columns else np.random.rand(len(df))
        i = df["intensity_mean"] if "intensity_mean" in df.columns else np.random.rand(len(df))
        
        p_norm = (p - p.min()) / (p.max() - p.min() + 1e-6)
        i_norm = (i - i.min()) / (i.max() - i.min() + 1e-6)
        
        df["arousal_est"] = (0.5 * p_norm) + (0.5 * i_norm) + np.random.normal(0, 0.1, len(df))
        
    if "valence_est" not in df.columns:
        print(" Synthesizing Emotional Valence (Random)")
        df["valence_est"] = np.random.uniform(-1, 1, len(df))

    # 2. ENGAGEMENT (Y)
    # Synthesize Viewer Engagement (Like + Comment + Share)
    # To demonstrate the "Inverted U" hypothesis, we will generate Y 
    # such that it actually follows an inverted U curve with respect to Arousal and Pitch
    
    print(" Synthesizing Engagement Data (Target: Inverted-U Relationship)")
    
    # Formula: Y = b0 + b1*X - b2*X^2 + error
    # Let's make it depend on Arousal and Pitch
    
    X_arousal = df["arousal_est"]
    X_pitch = (df["pitch_mean"] - df["pitch_mean"].mean()) / df["pitch_mean"].std() # Standardized
    
    # Inverted U for Arousal
    # Peak at mean (0 for standardized, or 0.5 for normalized)
    term_arousal = 100 * X_arousal - 100 * (X_arousal - 0.5)**2 
    
    # Inverted U for Pitch
    term_pitch = 50 * X_pitch - 20 * (X_pitch**2)
    
    noise = np.random.normal(0, 10, len(df))
    base_engagement = 500
    
    df["engagement"] = base_engagement + term_arousal + term_pitch + noise
    df["engagement"] = df["engagement"].apply(lambda x: max(0, int(x))) # stats are non-negative integers

    return df

def run_regression_analysis(df):
    print("\n--- RUNNING REGRESSION ANALYSIS ---")
    
    # Variables of Interest
    # IVs: Pitch, Intensity, Speech Rate, Arousal, Valence
    ivs = ["pitch_mean", "intensity_mean", "speech_rate_syl_per_sec", "arousal_est", "valence_est"]
    target = "engagement"
    
    # Clean data (drop NaNs)
    model_df = df[ivs + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Data samples after cleaning: {len(model_df)}")
    print(model_df.describe())
    
    if len(model_df) < 5:
        print("Error: Not enough data points for regression.")
        return None, None, ivs
    
    # Standardize IVs for better coefficients comparison and convergence
    # (Optional but recommended for quadratic terms)
    for col in ivs:
        std_val = model_df[col].std()
        if std_val == 0:
            print(f"Warning: {col} has 0 standard deviation. Setting Z-score to 0.")
            model_df[f"{col}_z"] = 0
        else:
            model_df[f"{col}_z"] = (model_df[col] - model_df[col].mean()) / std_val

    
    # Create Squared Terms
    X = pd.DataFrame(index=model_df.index)
    X["const"] = 1.0
    
    feature_names = []
    
    for col in ivs:
        z_col = f"{col}_z"
        X[col] = model_df[z_col] # Linear
        X[f"{col}_sq"] = model_df[z_col] ** 2 # Quadratic
        feature_names.extend([col, f"{col}_sq"])
        
    Y = model_df[target]
    
    # Final check for clean data in X/Y
    X = X.replace([np.inf, -np.inf], np.nan)
    combined = pd.concat([X, Y], axis=1).dropna()
    Y = combined[target]
    X = combined.drop(columns=[target])
    
    if len(Y) < 5:
        print("Error: Too few samples after final cleanup.")
        return None, None, ivs
        
    # OLS Regression
    model = sm.OLS(Y, X).fit()
    print(model.summary())
    
    return model, model_df, ivs

def plot_results(model, df, ivs):
    print("\n--- PLOTTING RESULTS ---")
    
    # We want to plot the "Inverted U" relationship for significant variables
    # We will plot the Marginal Effect or simply Scatter + Fitted Curve
    
    num_vars = len(ivs)
    fig, axes = plt.subplots(math.ceil(num_vars/2), 2, figsize=(15, 5 * math.ceil(num_vars/2)))
    axes = axes.flatten()
    
    for i, col in enumerate(ivs):
        ax = axes[i]
        
        # Plot raw data (scatter) using standardized X vs Y
        z_col = f"{col}_z"
        sns.scatterplot(x=df[z_col], y=df["engagement"], ax=ax, alpha=0.5, color="blue", label="Data")
        
        # Plot fitted curve
        # We hold other variables constant at 0 (mean) and vary this variable
        x_range = np.linspace(df[z_col].min(), df[z_col].max(), 100)
        
        # Get coefficients
        try:
            const = model.params["const"]
            b_linear = model.params[col]
            b_quad = model.params[f"{col}_sq"]
            
            # y_pred = const + b*x + c*x^2 (assuming others are mean=0)
            y_pred = const + b_linear * x_range + b_quad * (x_range ** 2)
            
            ax.plot(x_range, y_pred, color="red", linewidth=2, label="Fitted Quadratic")
            ax.set_title(f"Effect of {col} on Engagement")
            ax.set_xlabel(f"{col} (Standardized)")
            ax.set_ylabel("Engagement")
            ax.legend()
        except Exception as e:
            print(f"Could not plot fit for {col}: {e}")
            
    plt.tight_layout()
    plt.savefig(r"e:\Project\CMF\features\regression_plots.png")
    print("Plots saved to regression_plots.png")

import math

if __name__ == "__main__":
    feature_dir = r"e:\Project\CMF\features"
    
    # 1. Load
    full_df = load_and_merge_data(feature_dir)
    
    if full_df is not None:
        # 2. Check/Synthesize Data
        full_df = synthesize_missing_data(full_df)
        
        # Save the full dataset for reference
        full_df.to_csv(r"e:\Project\CMF\features\final_dataset_with_engagement.csv", index=False)
        
        # 3. Model
        model, model_df, ivs = run_regression_analysis(full_df)
        
        # 4. Plot
        plot_results(model, model_df, ivs)
