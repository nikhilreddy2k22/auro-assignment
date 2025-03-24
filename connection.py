import os
import psycopg2

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

DB_CONFIG = {
    "dbname": DBNAME,
    "user": USER,
    "password": PASSWORD,
    "host": HOST,
    "port": PORT,
}
