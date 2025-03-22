import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import pandas as pd

# Create base class for declarative models
Base = declarative_base()

# Define the anomaly data model
class AnomalyRecord(Base):
    __tablename__ = 'anomaly_records'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    time_value = Column(Integer)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    network_activity = Column(Float)
    is_anomaly = Column(Boolean)
    
    def __repr__(self):
        return f"<AnomalyRecord(id={self.id}, timestamp={self.timestamp}, is_anomaly={self.is_anomaly})>"

# Database connection function
def get_database_connection(db_type='sqlite'):
    """
    Create a database connection
    db_type: 'sqlite' (default, file-based) or 'mysql' (server-based)
    """
    if db_type == 'sqlite':
        # SQLite - simple file-based database
        engine = create_engine('sqlite:///anomaly_detection.db')
    elif db_type == 'mysql':
        # MySQL - requires separate MySQL server
        # Update with your MySQL credentials
        user = 'root'
        password = 'your_password'
        host = 'localhost'
        database = 'anomaly_detection'
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
    else:
        raise ValueError("Unsupported database type")
    
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Create session factory
    Session = sessionmaker(bind=engine)
    return Session()

def save_batch_to_database(batch_df, session=None):
    """
    Save a batch of data to the database
    batch_df: DataFrame with the current batch
    session: SQLAlchemy session (if None, a new session will be created)
    """
    close_session = False
    if session is None:
        session = get_database_connection()
        close_session = True
    
    try:
        # Convert DataFrame rows to AnomalyRecord objects
        records = []
        for _, row in batch_df.iterrows():
            # Check if 'predicted_anomaly' column exists, otherwise use 'anomaly' or default to False
            is_anomaly = False
            if 'predicted_anomaly' in row:
                is_anomaly = bool(row['predicted_anomaly'])
            elif 'anomaly' in row:
                is_anomaly = bool(row['anomaly'])
            
            record = AnomalyRecord(
                time_value=int(row['time']),
                cpu_usage=float(row['cpu_usage']),
                memory_usage=float(row['memory_usage']),
                network_activity=float(row['network_activity']),
                is_anomaly=is_anomaly
            )
            records.append(record)
        
        # Add all records to database
        session.add_all(records)
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        raise e
    finally:
        if close_session:
            session.close()

def get_historical_anomalies(hours=24, session=None):
    """
    Retrieve historical anomalies from the database
    hours: Number of hours to look back
    session: SQLAlchemy session (if None, a new session will be created)
    """
    close_session = False
    if session is None:
        session = get_database_connection()
        close_session = True
    
    try:
        # Calculate the time threshold
        time_threshold = datetime.datetime.utcnow() - datetime.timedelta(hours=hours)
        
        # Query anomalies since the threshold
        anomalies = (session.query(AnomalyRecord)
                    .filter(AnomalyRecord.timestamp >= time_threshold)
                    .filter(AnomalyRecord.is_anomaly == True)
                    .all())
        
        # Convert to DataFrame for easier handling in streamlit
        if anomalies:
            data = [{
                'timestamp': record.timestamp,
                'time_value': record.time_value,
                'cpu_usage': record.cpu_usage,
                'memory_usage': record.memory_usage,
                'network_activity': record.network_activity
            } for record in anomalies]
            return pd.DataFrame(data)
        return pd.DataFrame()
        
    finally:
        if close_session:
            session.close()

# Test code that only runs when database.py is executed directly
if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    session = get_database_connection()
    print("Connection established successfully!")
    
    # Create a test record
    test_record = AnomalyRecord(
        time_value=1000,
        cpu_usage=0.75,
        memory_usage=0.85,
        network_activity=0.25,
        is_anomaly=True
    )
    
    # Add and commit
    session.add(test_record)
    session.commit()
    print(f"Test record added: {test_record}")
    
    # Query and display
    records = session.query(AnomalyRecord).all()
    print(f"Total records in database: {len(records)}")
    
    # Close connection
    session.close()
    print("Database test completed.")