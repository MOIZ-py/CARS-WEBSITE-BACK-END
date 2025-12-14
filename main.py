# ===================== IMPORTS =====================
import mysql.connector
from mysql.connector import Error
import pandas as pd
import joblib

# ===================== DATABASE CONFIG =====================
DB_CONFIG = {
    "host": "localhost",
    "database": "cars",
    "user": "root",
    "password": "ABDULMOIZ"
}

# ===================== DATABASE UTILITIES =====================
def get_db_price_range(conn):
    cur = conn.cursor()
    cur.execute("SELECT MIN(PRICE), MAX(PRICE) FROM CARS")
    min_p, max_p = cur.fetchone()
    cur.close()
    return float(min_p), float(max_p)

# ===================== INPUT NORMALIZATION =====================
def normalize_inputs(s):
    if s["FUEL"] == "GAS":
        s["FUEL"] = "PETROL"

    if s["TYPE"] == "JEEP":
        s["TYPE"] = "SUV"

    if s["PURPOSE"] == "RACING":
        s["PURPOSE"] = "SPORTY"
    elif s["PURPOSE"] == "LONGTRIPS":
        s["PURPOSE"] = "LONG DRIVES"
    elif s["PURPOSE"] == "OFFROAD":
        s["PURPOSE"] = "OFF-ROAD"

    return s

# ===================== USER INPUT =====================
def get_user_input(min_price, max_price):
    s = {}

    s["BRAND"] = input("Brand (or ANY): ").strip().upper()
    if s["BRAND"] == "ANY":
        s["BRAND"] = None

    while True:
        try:
            s["BUDGET"] = float(input(f"Budget ({min_price:.0f}-{max_price:.0f}): "))
            if min_price <= s["BUDGET"] <= max_price:
                break
        except ValueError:
            pass
        print("Invalid budget.")

    s["FUEL"] = input("Fuel (PETROL/DIESEL/ELECTRIC/HYBRID/ANY): ").strip().upper()
    if s["FUEL"] == "ANY":
        s["FUEL"] = None

    while True:
        try:
            s["SEATINGCAPACITY"] = int(input("Seating capacity: "))
            if 2 <= s["SEATINGCAPACITY"] <= 15:
                break
        except ValueError:
            pass
        print("Invalid seating.")

    s["TYPE"] = input("Type (SEDAN/SUV/HATCHBACK/SPORTS/PICKUP/ANY): ").strip().upper()
    if s["TYPE"] == "ANY":
        s["TYPE"] = None

    s["PURPOSE"] = input("Purpose (CITY/ECONOMICAL/SPORTY/OFF-ROAD/LONG DRIVES/ANY): ").strip().upper()
    if s["PURPOSE"] == "ANY":
        s["PURPOSE"] = None

    s["STATUS"] = input("Status (NEW/USED/ANY): ").strip().upper()
    if s["STATUS"] == "ANY":
        s["STATUS"] = None

    return normalize_inputs(s)

# ===================== WHERE CLAUSE BUILDER =====================
def build_where_clause(s, brand_mode="include"):
    conditions = []
    params = []

    # BRAND filter
    if s.get("BRAND"):
        if brand_mode == "include":
            conditions.append("BRAND = %s")
            params.append(s["BRAND"])
        elif brand_mode == "exclude":
            conditions.append("BRAND <> %s")
            params.append(s["BRAND"])

    # FUEL filter
    if s.get("FUEL"):
        conditions.append("FUEL = %s")
        params.append(s["FUEL"])

    # TYPE filter
    if s.get("TYPE"):
        conditions.append("TYPE = %s")
        params.append(s["TYPE"])

    # SEATINGCAPACITY filter
    if s.get("SEATINGCAPACITY"):
        conditions.append("SEATINGCAPACITY = %s")
        params.append(s["SEATINGCAPACITY"])

    # PURPOSE filter
    if s.get("PURPOSE"):
        if s["PURPOSE"] in ("CITY", "ECONOMICAL"):
            conditions.append("(PURPOSE = %s OR PURPOSE = 'FAMILY')")
            params.append(s["PURPOSE"])
        elif s["PURPOSE"] in ("SPORTY", "OFF-ROAD"):
            conditions.append("(PURPOSE = 'SPORTY' OR PURPOSE = 'OFF-ROAD')")
        else:
            conditions.append("PURPOSE = %s")
            params.append(s["PURPOSE"])

    # Build WHERE clause string
    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params

# ===================== PAGINATED QUERY =====================
def paginated_query(conn, base_sql, params, title):
    offset = 0
    limit = 20

    while True:
        sql = f"{base_sql} LIMIT {limit} OFFSET {offset}"
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

        if not rows:
            if offset == 0:
                print("\nNo cars found.")
            break

        print(f"\n--- {title} ---")
        print(pd.DataFrame(rows).to_string(index=False))

        if len(rows) < limit:
            break

        if input("\nMore results? (yes/no): ").strip().upper() != "YES":
            break

        offset += limit

# ===================== ML PREDICTION =====================
MODEL_PATH = "car_price_model.joblib"

def predict_price():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found.")
        return

    brand = input("Brand: ").strip().lower()
    fueltype = input("Fuel type(petrol,diesel,hybrid,electric): ").strip().lower()
    cartype = input("Car type(SEDAN,SUV,COUPE,HATCHBACK,CONVERTIBLE,CROSSOVER,PICKUP): ").strip().lower()
    status = input("Status (NEW/USED): ").strip().lower()

    while True:
        try:
            enginesize = float(input("Engine size : "))
            if 0.6 <= enginesize <= 5.0:
                break
        except ValueError:
            pass

    X = pd.DataFrame({
        "brand": [brand],
        "fueltype": [fueltype],
        "cartype": [cartype],
        "status": [status],
        "enginesize": [enginesize]
    })

    price = model.predict(X)[0]
    print(f"\nEstimated Price: ${price:,.2f}")

# ===================== MAIN MENU =====================
def main():
    print("\n===== Car Finder & Price Predictor =====")

    while True:
        print("\n1. Predict car price")
        print("2. Find cars from database")
        print("3. Exit")

        choice = input("Choose option: ").strip()

        if choice == "1":
            predict_price()

        elif choice == "2":
            try:
                conn = mysql.connector.connect(**DB_CONFIG)
            except Error as e:
                print("Database connection failed:", e)
                continue

            min_p, max_p = get_db_price_range(conn)
            s = get_user_input(min_p, max_p)

            # Function to safely append PRICE condition
            def append_price_clause(where_clause):
                if where_clause:
                    return where_clause + " AND `PRICE` <= %s"
                else:
                    return " WHERE `PRICE` <= %s"

            # Exact match
            where1, params1 = build_where_clause(s, "include")
            sql1 = f"SELECT * FROM CARS{append_price_clause(where1)}"
            paginated_query(conn, sql1, params1 + [s["BUDGET"]], "Exact Matches")

            # Other brands
            if s["BRAND"]:
                where2, params2 = build_where_clause(s, "exclude")
                sql2 = f"SELECT * FROM CARS{append_price_clause(where2)}"
                paginated_query(conn, sql2, params2 + [s["BUDGET"]], "Other Brand Options")

            # 20% higher budget
            where3, params3 = build_where_clause(s, "include")
            sql3 = f"SELECT * FROM CARS{append_price_clause(where3)}"
            paginated_query(conn, sql3, params3 + [s["BUDGET"] * 1.2], "20% Higher Budget")

            conn.close()

        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Invalid choice.")
if __name__ == "__main__":
    main()

