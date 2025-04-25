import streamlit as st
import torch
import torch.nn as nn
import torch.serialization
import bcrypt
import json
import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Model definition
class FraudDetector(nn.Module):
    def __init__(self, input_size):
        super(FraudDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Load model and preprocessors
input_size = 20
model = FraudDetector(input_size)
model.load_state_dict(torch.load('fraud_detector.pth'))
model.eval()

torch.serialization.add_safe_globals([StandardScaler, LabelEncoder])
scaler = torch.load('scaler.pth', weights_only=False)
label_encoders = torch.load('label_encoders.pth', weights_only=False)

# MySQL Connection
def get_db_connection():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="PasswoRd",
        port=3306,
        database="users",
        auth_plugin="mysql_native_password"
    )
    return db

# File processor
def extract_transaction_data(file_content):
    try:
        content = file_content.decode('utf-8')
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            transaction_data = {}
            for line in content.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    transaction_data[key.strip()] = value.strip()
            return transaction_data
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return None

def prepare_features(input_data):
    """
    Construct feature vector matching training pipeline and apply scaler.
    Pads or truncates to match scaler's expected input length.
    """
    try:
        features = []
        # Numeric features (must match training order)
        numeric_cols = [
            'account_balance', 'transaction_amount', 'previous_fraudulent_activity',
            'failed_transaction_count_7d', 'daily_transaction_count',
            'avg_transaction_amount_7d', 'card_age', 'risk_score'
        ]
        for col in numeric_cols:
            val = input_data.get(col, 0)
            try:
                features.append(float(val))
            except:
                features.append(0.0)
        # Categorical features via label encoding
        categorical_cols = [
            'device_type', 'merchant_category', 'authentication_method', 'card_type'
        ]
        for col in categorical_cols:
            encoder = label_encoders.get(col)
            raw = input_data.get(col, None)
            try:
                encoded = encoder.transform([raw])[0] if raw in encoder.classes_ else 0
            except:
                encoded = 0
            features.append(float(encoded))
        # Adjust feature length to match scaler
        expected = scaler.mean_.shape[0]
        actual = len(features)
        if actual < expected:
            # Pad with zeros
            features.extend([0.0] * (expected - actual))
        elif actual > expected:
            # Truncate extra features
            features = features[:expected]
        # Scale and return
        scaled = scaler.transform([features])
        return scaled[0]
    except Exception as e:
        st.error(f"Feature preparation error: {e}")
        return None


# UI Configuration
st.set_page_config(page_title="Online Banking Transactions", layout="wide")

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "User Authentication"

# Sidebar Navigation
st.sidebar.title("Navigation")
if not st.session_state.authenticated:
    st.sidebar.markdown("🔑 User Authentication")
    if st.sidebar.button("Go to Authentication"):
        st.session_state.current_page = "User Authentication"
    st.sidebar.markdown("🔒 Transaction Prediction")
    st.sidebar.markdown("🔒 Transaction History")
    st.sidebar.markdown("🔒 Transaction Graph")
else:
    if st.sidebar.button("🔑 User Authentication"):
        st.session_state.current_page = "User Authentication"
    if st.sidebar.button("🤖 Transaction Prediction"):
        st.session_state.current_page = "Transaction Prediction"
    if st.sidebar.button("💰 Transaction History"):
        st.session_state.current_page = "Transaction History"
    if st.sidebar.button("📈 Transaction Graph"):
        st.session_state.current_page = "Transaction Graph"

def login():
    with st.form("signin_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        sign_in_btn = st.form_submit_button("Sign In")

    if sign_in_btn:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, password FROM user_cl WHERE email = %s", (email,))
            user = cursor.fetchone()
            if user:
                stored_hash = user[1].encode("utf-8")
                if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
                    st.session_state.authenticated = True
                    st.session_state.username = user[0]
                    st.success(f"Welcome, {st.session_state.username}")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.error("User not found.")
        except Exception as e:
            st.error(f"DB error: {e}")
        finally:
            conn.close()

def signup():
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        sign_up_btn = st.form_submit_button("Sign Up")

    if sign_up_btn:
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM user_cl WHERE email = %s", (email,))
            if cursor.fetchone():
                st.warning("An account already exists with this email.")
            else:
                cursor.execute(
                    "INSERT INTO user_cl (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, password_hash)
                )
                conn.commit()
                st.success("Account created successfully. Please sign in.")
        except Exception as e:
            st.error(f"Registration error: {e}")
        finally:
            conn.close()

def transaction_predictor():
    st.markdown(f"<div style='text-align: right; font-weight: bold; font-size: 18px;'>Welcome, {st.session_state.username}</div>", unsafe_allow_html=True)
    st.title("🤖 Transaction Prediction")
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios as needed
    with col2:
        age = st.number_input("Enter your Age", min_value=1, max_value=120, step=1)
        salary = st.number_input("Enter your Salary", min_value=0.0, step=100.0, format="%.2f")

        if age > 19 and salary > 2000:
            st.subheader("Upload Transaction File")
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])

            if uploaded_file is not None and st.button("Submit for Prediction"):
                file_content = uploaded_file.read()
                data = extract_transaction_data(file_content)

                if data:
                    try:
                        numeric_features = prepare_features(data)
                        input_tensor = torch.tensor([numeric_features], dtype=torch.float32)
                        output = model(input_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                        label = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
                        st.success(f"Prediction: {label}")

                        try:
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO transactions (
                                    account_balance, device_type, transaction_amount, timestamp,
                                    merchant_category, previous_fraudulent_activity,
                                    failed_transaction_count_7d, authentication_method,
                                    daily_transaction_count, avg_transaction_amount_7d,
                                    card_type, card_age, risk_score, fraud_label
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                                (
                                    data.get('account_balance'),
                                    data.get('device_type'),
                                    data.get('transaction_amount'),
                                    data.get('timestamp', datetime.now().isoformat()),
                                    data.get('merchant_category'),
                                    data.get('previous_fraudulent_activity'),
                                    data.get('failed_transaction_count_7d'),
                                    data.get('authentication_method'),
                                    data.get('daily_transaction_count'),
                                    data.get('avg_transaction_amount_7d'),
                                    data.get('card_type'),
                                    data.get('card_age'),
                                    data.get('risk_score'),
                                    prediction
                                )
                            )
                            conn.commit()
                            st.success("Transaction saved successfully.")
                        except Exception as db_err:
                            st.error(f"Database insert failed: {db_err}")
                        finally:
                            conn.close()
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    st.error("Invalid file format or content.")

def transaction_history():
    st.title("💰 Transaction History")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Fetch all transactions for current user, remove exact duplicates in Python
        cursor.execute("SELECT DISTINCT transaction_amount, fraud_label FROM transactions")
        records = cursor.fetchone()
        if records:
            st.subheader("🕒 Latest Transaction")
            status = "Fraudulent Transaction" if records["fraud_label"] == 1 else "Legitimate Transaction"
            st.markdown(f"**Amount:** ${records['transaction_amount']:.2f}  \n**Status:** {status}")
        
        if records:
            table_data = []
            seen = set()
            for rec in records:
                key = (rec["transaction_amount"], rec["fraud_label"])
                if key not in seen:
                    seen.add(key)
                    status = "Fraudulent Transaction" if rec["fraud_label"] == 1 else "Legitimate Transaction"
                    table_data.append({
                        "Transaction ID": len(table_data) + 1,  
                        "Transaction Amount": f"${rec['transaction_amount']:.2f}",
                        "Status": status
                    })

            st.table(table_data)
        else:
            st.info("No transactions found.")
    except Exception as e:
        st.error(f"Error loading history: {e}")
    finally:
        conn.close()

def transaction_graph():
    st.title("📈 Transaction Graph ")

    try:
        # Get data from DB
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT timestamp, transaction_amount, avg_transaction_amount_7d FROM transactions")
        records = cursor.fetchall()
        conn.close()

        if not records:
            st.info("No transaction data available.")
            return

        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # --- Date Inputs ---
        st.subheader("Enter Date Range (dd-mm-yy)")
        col1, col2 = st.columns(2)

        with col1:
            start_date_str = st.text_input("Start Date (dd-mm-yy)", value=datetime.now().strftime("%d-%m-%y"))
        with col2:
            end_date_str = st.text_input("End Date (dd-mm-yy)", value=datetime.now().strftime("%d-%m-%y"))

        try:
            start_date = datetime.strptime(start_date_str, "%d-%m-%y")
            end_date = datetime.strptime(end_date_str, "%d-%m-%y")

            if start_date > end_date:
                st.warning("Start date must be before end date.")
                return

            # Filter data
            filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

            if filtered_df.empty:
                st.info("No data found for the selected date range.")
                return

            # Format X-axis as mm-yy
            filtered_df['period'] = filtered_df['timestamp'].dt.strftime('%m-%y')

            # Group by month-year
            grouped = filtered_df.groupby('period').agg({
                'transaction_amount': 'sum',
                'avg_transaction_amount_7d': 'mean'
            }).reset_index()

            # Sort by actual datetime for proper order
            grouped['period_dt'] = pd.to_datetime(grouped['period'], format='%m-%y')
            grouped = grouped.sort_values('period_dt')

            # Plot
            fig = px.line(
                grouped,
                x='period',
                y=['transaction_amount', 'avg_transaction_amount_7d'],
                labels={'value': 'Amount', 'period': 'Month-Year'},
                title=f"Transactions from {start_date_str} to {end_date_str}"
            )

            fig.update_layout(
                xaxis_title="Month-Year",
                yaxis_title="Amount",
                xaxis_tickangle=45,
                xaxis=dict(type='category')  
            )

            st.plotly_chart(fig, use_container_width=True)

        except ValueError:
            st.error("Invalid date format. Please enter in dd-mm-yy.")

    except Exception as e:
        st.error(f"Error loading data: {e}")

# Page Routing
if st.session_state.current_page == "User Authentication":
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        st.subheader("User Authentication")
        auth_mode = st.radio("Choose Authentication Mode", ["Sign In", "Sign Up"], horizontal=True)
        if auth_mode == "Sign In":
            login()
        else:
            signup()

elif st.session_state.current_page == "Transaction Prediction" and st.session_state.authenticated:
    transaction_predictor()
elif st.session_state.current_page == "Transaction History" and st.session_state.authenticated:
    transaction_history()
elif st.session_state.current_page == "Transaction Graph" and st.session_state.authenticated:
    transaction_graph()
