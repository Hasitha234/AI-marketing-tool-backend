# pricing_utils.py
import numpy as np

def simulate_route_with_constraints(route_df, demand_model, feature_cols, cap=0.2):
    """
    Simulate demand + revenue across a price range for a single route,
    applying price caps to avoid extreme jumps.
    """
    if route_df.empty:
        return None

    # Features must include 'fare'
    X_route = route_df[feature_cols + ['fare']]
    avg_price = route_df['fare'].mean()
    price_range = np.linspace(avg_price * (1 - cap), avg_price * (1 + cap), 20)

    revenues = []
    for p in price_range:
        X_test = X_route.copy()
        X_test['fare'] = p
        demand_pred = demand_model.predict(X_test).mean()
        revenue = p * demand_pred
        revenues.append(revenue)

    best_price = price_range[np.argmax(revenues)]

    return {
        "route": f"{route_df['city1'].iloc[0]} → {route_df['city2'].iloc[0]}",
        "avg_price": avg_price,
        "best_price": best_price,
        "current_revenue": avg_price * demand_model.predict(X_route).mean(),
        "optimal_revenue": max(revenues),
        "uplift_%": (max(revenues) - avg_price * demand_model.predict(X_route).mean()) 
                     / (avg_price * demand_model.predict(X_route).mean()) * 100
    }


def pricing_policy(route_df, demand_model, feature_cols, cap=0.2):
    """
    Apply business rules to the raw simulation output
    (rounding, caps, uplift sanity checks).
    """
    res = simulate_route_with_constraints(route_df, demand_model, feature_cols, cap=cap)
    if res is None:
        return None

    rec_price = res["best_price"]

    # Round to nearest $5
    rec_price = round(rec_price / 5) * 5

    # Prevent unrealistic jumps (>30% uplift)
    if res["uplift_%"] > 30:
        rec_price = res["avg_price"] * (1 + cap)

    res["recommended_price"] = rec_price
    return res
