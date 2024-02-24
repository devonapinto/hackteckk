from galucoma import *
from DR import  *
from cataract import *
conn = psycopg2.connect(
    dbname="eyes",
    user="postgres",
    password="Maqwin@95",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()
cursor.execute("DELETE FROM result;")
conn.commit()
cursor.close()
conn.close()
glaucoma()
DR()
cat()

