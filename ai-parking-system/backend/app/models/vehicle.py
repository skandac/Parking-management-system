"""
Vehicle model for the AI Parking Management System
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class Vehicle(Base):
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(20), unique=True, index=True, nullable=False)
    vehicle_type = Column(String(20), default="car")  # car, truck, motorcycle, bus
    
    # AI Detection Fields
    detection_confidence = Column(Float, default=0.0)
    detection_timestamp = Column(DateTime, default=func.now())
    detection_method = Column(String(20), default="yolo")  # yolo, haar, manual
    
    # Vehicle Properties
    make = Column(String(50))
    model = Column(String(50))
    color = Column(String(30))
    year = Column(Integer)
    
    # Dimensions
    width = Column(Float)
    length = Column(Float)
    height = Column(Float)
    
    # Current Status
    current_slot_id = Column(Integer, nullable=True)
    entry_time = Column(DateTime, default=func.now())
    exit_time = Column(DateTime, nullable=True)
    
    # User Association
    user_id = Column(Integer, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    notes = Column(Text)
    
    def __repr__(self):
        return f"<Vehicle(id={self.id}, license_plate='{self.license_plate}', type={self.vehicle_type})>"
    
    @property
    def is_parked(self):
        """Check if vehicle is currently parked"""
        return self.current_slot_id is not None and self.exit_time is None
    
    @property
    def parking_duration(self):
        """Calculate current parking duration in minutes"""
        if self.entry_time and not self.exit_time:
            return (datetime.utcnow() - self.entry_time).total_seconds() / 60
        return 0
