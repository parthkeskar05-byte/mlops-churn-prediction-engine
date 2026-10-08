"""Generate synthetic telco churn dataset."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_telco_churn_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic telco customer churn dataset.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with synthetic telco churn data
    """
    np.random.seed(random_state)

    # Customer demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])

    # Account information
    tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)

    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58]),
        'No phone service'
    )

    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])

    online_security = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.29, 0.71]),
        'No internet service'
    )

    online_backup = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.34, 0.66]),
        'No internet service'
    )

    device_protection = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.34, 0.66]),
        'No internet service'
    )

    tech_support = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.29, 0.71]),
        'No internet service'
    )

    streaming_tv = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.38, 0.62]),
        'No internet service'
    )

    streaming_movies = np.where(
        internet_service != 'No',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.39, 0.61]),
        'No internet service'
    )

    # Billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.33, 0.23, 0.22, 0.22]
    )

    monthly_charges = np.random.uniform(18.25, 118.75, n_samples)
    total_charges = monthly_charges * tenure

    # Generate churn with realistic correlations
    churn_prob = 0.27  # Base churn rate

    # Adjust probability based on features
    churn_score = np.zeros(n_samples)
    churn_score += (senior_citizen == 1) * 0.15
    churn_score += (partner == 'No') * 0.10
    churn_score += (dependents == 'No') * 0.08
    churn_score += (tenure < 12) * 0.25
    churn_score += (contract == 'Month-to-month') * 0.30
    churn_score += (internet_service == 'Fiber optic') * 0.12
    churn_score += (online_security == 'No') * 0.08
    churn_score += (tech_support == 'No') * 0.08
    churn_score += (payment_method == 'Electronic check') * 0.10
    churn_score += (monthly_charges > 70) * 0.10

    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = (np.random.random(n_samples) < churn_probability).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges.round(2),
        'TotalCharges': total_charges.round(2),
        'Churn': churn
    })

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "telco_churn.csv"
    df = generate_telco_churn_data()
    df.to_csv(output_path, index=False)
    print(f"✓ Generated {len(df)} samples at {output_path}")
    print(f"✓ Churn rate: {df['Churn'].mean():.1%}")
