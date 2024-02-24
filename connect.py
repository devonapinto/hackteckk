import psycopg2

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    dbname="eyes",
    user="postgres",
    password="Maqwin@95",
    host="localhost",
    port="5432"
)

# Create a cursor object using the cursor() method
cursor = conn.cursor()

# Insert a row of data
cursor.execute("INSERT INTO result (disorders,final) VALUES ('gl','true');")

# Save (commit) the changes
conn.commit()

# Retrieve data
cursor.execute("SELECT * FROM result;")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()
