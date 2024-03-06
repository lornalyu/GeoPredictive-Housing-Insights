import sqlite3

# Create a new database in memory
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Define the SQL commands to create the tables
commands = [
    '''CREATE TABLE Company (
        companyID INTEGER PRIMARY KEY, 
        cname TEXT, 
        address TEXT, 
        city TEXT, 
        state TEXT, 
        zipcode TEXT
    )''',
    '''CREATE TABLE Computer (
        computerID INTEGER PRIMARY KEY, 
        model TEXT, 
        laptop BOOLEAN, 
        os TEXT, 
        speed REAL, 
        ram INTEGER, 
        harddrive INTEGER, 
        screen TEXT, 
        price REAL
    )''',
    '''CREATE TABLE ProduceComputer (
        companyID INTEGER, 
        computerID INTEGER, 
        FOREIGN KEY (companyID) REFERENCES Company(companyID),
        FOREIGN KEY (computerID) REFERENCES Computer(computerID)
    )''',
    '''CREATE TABLE Printer (
        printerID INTEGER PRIMARY KEY, 
        model TEXT, 
        color BOOLEAN, 
        price REAL, 
        scanner BOOLEAN, 
        wireless BOOLEAN
    )''',
    '''CREATE TABLE ProducePrinter (
        companyID INTEGER, 
        printerID INTEGER,
        FOREIGN KEY (companyID) REFERENCES Company(companyID),
        FOREIGN KEY (printerID) REFERENCES Printer(printerID)
    )'''
]

# Execute the SQL commands to create the tables
for command in commands:
    cursor.execute(command)

# Commit the changes
conn.commit()

# Display a message
print("Database and tables created successfully.")

# Close the connection if you are done with it.
# Be sure any changes have been committed or they will be lost.
conn.close()
