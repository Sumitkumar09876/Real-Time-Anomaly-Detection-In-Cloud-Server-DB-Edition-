"""
Database management utilities for the anomaly detection system
"""
import argparse
from sqlalchemy import func
from database import get_database_connection, AnomalyRecord
import pandas as pd
import datetime

def export_data(output_file, days=None):
    """Export data from database to CSV file"""
    session = get_database_connection()
    try:
        query = session.query(AnomalyRecord)
        
        if days:
            time_threshold = datetime.datetime.utcnow() - datetime.timedelta(days=days)
            query = query.filter(AnomalyRecord.timestamp >= time_threshold)
        
        records = query.all()
        
        # Convert to DataFrame and save to CSV
        data = [{
            'id': record.id,
            'timestamp': record.timestamp,
            'time_value': record.time_value,
            'cpu_usage': record.cpu_usage,
            'memory_usage': record.memory_usage,
            'network_activity': record.network_activity,
            'is_anomaly': record.is_anomaly
        } for record in records]
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} records to {output_file}")
    
    finally:
        session.close()

def cleanup_old_data(days=30):
    """Remove data older than specified days"""
    session = get_database_connection()
    try:
        time_threshold = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        result = session.query(AnomalyRecord).filter(AnomalyRecord.timestamp < time_threshold).delete()
        session.commit()
        print(f"Removed {result} records older than {days} days")
        
    except Exception as e:
        session.rollback()
        print(f"Error: {str(e)}")
    finally:
        session.close()

def show_stats():
    """Show basic database statistics"""
    session = get_database_connection()
    try:
        total_count = session.query(func.count(AnomalyRecord.id)).scalar()
        anomaly_count = session.query(func.count(AnomalyRecord.id)).filter(AnomalyRecord.is_anomaly == True).scalar()
        
        if total_count > 0:
            oldest_record = session.query(func.min(AnomalyRecord.timestamp)).scalar()
            newest_record = session.query(func.max(AnomalyRecord.timestamp)).scalar()
            
            print(f"Database Statistics:")
            print(f"- Total records: {total_count}")
            print(f"- Anomalies: {anomaly_count} ({anomaly_count/total_count*100:.2f}%)")
            print(f"- Date range: {oldest_record} to {newest_record}")
        else:
            print("Database is empty")
    
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database management utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data to CSV")
    export_parser.add_argument("output", help="Output CSV file")
    export_parser.add_argument("--days", type=int, help="Export only data from last N days")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove old data")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Remove data older than N days")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if args.command == "export":
        export_data(args.output, args.days)
    elif args.command == "cleanup":
        cleanup_old_data(args.days)
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()