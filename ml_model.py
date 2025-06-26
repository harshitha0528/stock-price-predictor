'''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(df):
    # Features = today's close price
    X = df[['Close']]  # must be 2D

    # Labels = tomorrow's close price
    y = df['Prediction']  # 1D

    # Split into train and test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict using test data
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(df):
    # Features = today's close price
    X = df[['Close']]
    y = df['Prediction']

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Accuracy metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

    return model, X_test, y_test, predictions, metrics
