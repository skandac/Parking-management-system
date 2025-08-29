"""
Parking slot model for the AI Parking Management System
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class ParkingSlot(Base):
    __tablename__ = "parking_slots"
    
    id = Column(Integer, primary_key=True, index=True)
    slot_number = Column(String(10), unique=True, index=True, nullable=False)
    floor_level = Column(Integer, default=1)
    section = Column(String(50))
    slot_type = Column(String(20), default="standard")  # standard, disabled, electric, premium
    is_available = Column(Boolean, default=True)
    is_reserved = Column(Boolean, default=False)
    current_vehicle_id = Column(Integer, nullable=True)
    
    # AI Detection Fields
    confidence_score = Column(Float, default=0.0)
    last_detection_time = Column(DateTime, default=func.now())
    detection_status = Column(String(20), default="empty")  # empty, occupied, reserved
    
    # Physical Properties
    width = Column(Float, default=2.5)  # meters
    length = Column(Float, default=5.0  # meters
    height_limit = Column(Float, default=2.2)  # meters
    
    # Location
    x_coordinate = Column(Float)
    y_coordinate = Column(Float)
    zone = Column(String(50))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    notes = Column(Text)
    
    def __repr__(self):
        return f"<ParkingSlot(id={self.id}, slot_number='{self.slot_number}', available={self.is_available})>"
    
    @property
    def status(self):
        """Get current status of the parking slot"""
        if self.is_reserved:
            return "reserved"
        elif not self.is_available:
            return "occupied"
        else:
            return "available"
