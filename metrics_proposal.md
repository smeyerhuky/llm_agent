
The user wants me to enhance my previous response about the metrics and monitoring system in two ways:

1. Add detailed comments to the same code I provided earlier, explaining what each part of the metrics service does and how it connects together
2. Create a README file that serves as an onboarding guide for new engineers, explaining:
   - Chain of thought concept
   - Code samples for getting started
   - Explanations of technology choices (like why InfluxDB, alternatives, etc.)
   - Documentation links
   - UX endpoints and imports
   - FAQ
   - Research topics
   - Strategy guide for onboarding

I should aim this at engineers who might be stumped by the technical details I provided earlier, making the system more accessible.

I'll start by organizing my response with the same artifacts, but with extensive comments added to explain the code. Then I'll create a comprehensive README file that addresses all the requirements for onboarding new engineers.

For the README, I'll need to:
1. Explain the overall purpose and architecture of the metrics system
2. Discuss technology choices (InfluxDB vs. alternatives like DuckDB, MongoDB, etc.)
3. Provide clear getting started examples
4. Include a FAQ section
5. Add research links
6. Create a strategy guide for onboarding
7. Include documentation for key UX endpoints

Let me do this while maintaining the same overall structure, just enhancing it with better documentation.

# Metrics and Monitoring System for LLM Agent

I'll design a complete metrics and monitoring system that will help you analyze performance, execution spans, debug-level traces, and system statistics for your LLM agent. This solution will follow a similar architecture to your current system while adding dedicated monitoring capabilities.

## Overall Architecture

Let's create a comprehensive metrics system with these components:

1. **Metrics Collector**: Python service that captures spans, traces, and timing data
2. **Time-Series Database**: InfluxDB for storing performance metrics
3. **Trace Storage**: MongoDB for storing execution spans and trace information
4. **API Gateway**: FastAPI service for metrics submission and querying
5. **Dashboard**: React frontend to visualize metrics and traces
6. **Log Aggregator**: Collect and centralize logs from all services

Let me provide the code for each component:

# Docker Compose file for the metrics and monitoring stack
# This orchestrates all the required services for our monitoring solution

version: '3.8'

services:
  # InfluxDB - Time series database for storing metrics data
  # We use InfluxDB because it's optimized for time-series data and provides
  # built-in aggregation functions, retention policies, and continuous queries
  influxdb:
    image: influxdb:2.0
    container_name: metrics_influxdb
    ports:
      - "8086:8086"
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup           # Auto-setup the database on first run
      - DOCKER_INFLUXDB_INIT_USERNAME=admin       # Admin username
      - DOCKER_INFLUXDB_INIT_PASSWORD=adminpassword  # Admin password (use secrets in production)
      - DOCKER_INFLUXDB_INIT_ORG=llm_agent_org    # Organization name
      - DOCKER_INFLUXDB_INIT_BUCKET=metrics       # Default bucket (database) name
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=your-super-secret-influx-token  # API token
    volumes:
      - influxdb_data:/var/lib/influxdb2          # Persistent storage for database files
    restart: unless-stopped
    networks:
      - metrics_network
      
  # MongoDB - Document database for storing trace and span information
  # We use MongoDB for trace data because:
  # 1. It provides flexible schema for evolving data models
  # 2. Good query performance for document-oriented data
  # 3. Easy scaling for high write throughput
  mongodb:
    image: mongo:latest
    container_name: metrics_mongodb
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin           # Root username
      - MONGO_INITDB_ROOT_PASSWORD=adminpassword   # Root password (use secrets in production)
    volumes:
      - mongodb_data:/data/db                      # Persistent storage for database files
    restart: unless-stopped
    networks:
      - metrics_network
  
  # Metrics Collector Service - Receives and processes metrics data
  # This service provides endpoints for collecting metrics, logs, spans, and traces
  # It batches data and writes it to the appropriate databases
  metrics_collector:
    build:
      context: ./metrics_services
      dockerfile: Dockerfile.collector
    container_name: metrics_collector
    ports:
      - "5001:5001"
    environment:
      - INFLUXDB_URL=http://influxdb:8086           # InfluxDB connection settings
      - INFLUXDB_TOKEN=your-super-secret-influx-token
      - INFLUXDB_ORG=llm_agent_org
      - INFLUXDB_BUCKET=metrics
      - MONGODB_URI=mongodb://admin:adminpassword@mongodb:27017/  # MongoDB connection
      - MONGODB_DB=agent_traces                      # MongoDB database name
      - LOG_LEVEL=INFO                               # Logging verbosity
    depends_on:
      - influxdb
      - mongodb
    restart: unless-stopped
    networks:
      - metrics_network
  
  # API Gateway - Provides a unified API for querying metrics and traces
  # This service offers endpoints for building dashboards and tools
  # It abstracts the underlying databases and provides higher-level query functions
  api_gateway:
    build:
      context: ./metrics_services
      dockerfile: Dockerfile.api
    container_name: metrics_api
    ports:
      - "8000:8000"
    environment:
      - INFLUXDB_URL=http://influxdb:8086           # InfluxDB connection settings 
      - INFLUXDB_TOKEN=your-super-secret-influx-token
      - INFLUXDB_ORG=llm_agent_org
      - INFLUXDB_BUCKET=metrics
      - MONGODB_URI=mongodb://admin:adminpassword@mongodb:27017/  # MongoDB connection
      - MONGODB_DB=agent_traces                      # MongoDB database name
      - COLLECTOR_URL=http://metrics_collector:5001  # Collector service URL
      - LOG_LEVEL=INFO                               # Logging verbosity
    depends_on:
      - influxdb
      - mongodb
      - metrics_collector
    restart: unless-stopped
    networks:
      - metrics_network
  
  # Dashboard frontend - React-based web interface
  # Provides visualizations, filtering, and exploration of metrics data
  # Built with React and Bootstrap for responsive design
  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    container_name: metrics_dashboard
    ports:
      - "3000:3000"                                  # Web interface port
    environment:
      - REACT_APP_API_URL=http://localhost:8000      # API Gateway URL (from browser perspective)
    depends_on:
      - api_gateway
    restart: unless-stopped
    networks:
      - metrics_network
  
  # Log aggregator service using Graylog
  # Collects, indexes and makes logs searchable from all services
  # Provides a web interface for log exploration and alerts
  graylog:
    image: graylog/graylog:4.3
    container_name: metrics_graylog
    environment:
      - GRAYLOG_PASSWORD_SECRET=somepasswordpepper   # Used for password encryption
      - GRAYLOG_ROOT_PASSWORD_SHA2=8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918  # "admin" hashed
      - GRAYLOG_HTTP_EXTERNAL_URI=http://localhost:9000/  # Public URL
      - GRAYLOG_MONGODB_URI=mongodb://admin:adminpassword@mongodb:27017/graylog  # MongoDB for Graylog
      - GRAYLOG_ELASTICSEARCH_HOSTS=http://elasticsearch:9200  # Elasticsearch connection
    ports:
      - "9000:9000"     # Web interface
      - "12201:12201/udp"  # GELF UDP input for log collection
      - "1514:1514"     # Syslog input
    depends_on:
      - mongodb
      - elasticsearch
    networks:
      - metrics_network
  
  # Elasticsearch for Graylog - Used for log indexing and search
  # Enables full-text search capabilities for the logging system
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.10.2
    container_name: metrics_elasticsearch
    environment:
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"  # Memory limits for Elasticsearch JVM
      - "discovery.type=single-node"       # Run as a single node (not a cluster)
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data  # Persistent storage
    networks:
      - metrics_network

# Persistent volumes to store data between container restarts
volumes:
  influxdb_data:     # Stores time-series metrics data
  mongodb_data:      # Stores trace data and span information
  elasticsearch_data:  # Stores log indices

# Network configuration for service communication
networks:
  metrics_network:
    driver: bridge   # Standard bridge network for container communication

Now, let's create the metrics collector service that will capture spans, traces, and performance data:

#!/usr/bin/env python3
"""
Metrics Collector Service for LLM Agent
---------------------------------------
Collects metrics, spans, and traces from the LLM agent system and stores them
in appropriate databases (InfluxDB for metrics, MongoDB for spans/traces).

This service provides the foundation for:
1. Performance monitoring (execution times, throughput)
2. Distributed tracing (tracking request flow across services)
3. Debug information capture (variable states, execution paths)
4. Log centralization
"""

import os
import time
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# InfluxDB for metrics - efficient time-series database with built-in
# aggregation functions and time-based queries
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# MongoDB for traces and spans - flexible document store for
# hierarchical and variably structured trace data
import motor.motor_asyncio

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("metrics-collector")

# Initialize FastAPI app - provides our HTTP endpoints with automatic
# OpenAPI documentation and request validation
app = FastAPI(title="LLM Agent Metrics Collector")

# Add CORS middleware to allow cross-origin requests from the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables - makes the service configurable
# without code changes and supports different deployment environments
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-super-secret-influx-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "llm_agent_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "agent_traces")

# Initialize InfluxDB client - for storing performance metrics and numeric data
influxdb_client = InfluxDBClient(
    url=INFLUXDB_URL, 
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG
)
influx_write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)

# Initialize MongoDB client - for storing trace data, logs, and debug information
mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
mongodb_database = mongodb_client[MONGODB_DB]
spans_collection = mongodb_database["spans"]
traces_collection = mongodb_database["traces"]
logs_collection = mongodb_database["logs"]
variables_collection = mongodb_database["variables"]

# Pydantic models for API request validation
# These define the data structures for our API and enforce type checking
class MetricData(BaseModel):
    """
    Represents a single metric data point with a name, value, and optional metadata.
    Metrics are typically numeric values that change over time (like execution duration).
    """
    name: str  # Metric name (e.g., "function.duration", "memory.usage")
    value: float  # The measured value
    timestamp: Optional[int] = None  # Unix timestamp in milliseconds
    tags: Dict[str, str] = Field(default_factory=dict)  # Metadata for filtering/grouping

class SpanData(BaseModel):
    """
    Represents a single operation or unit of work in a trace.
    Spans form a hierarchical structure showing the execution path.
    """
    span_id: str  # Unique identifier for this span
    trace_id: str  # Which trace this span belongs to
    parent_span_id: Optional[str] = None  # Parent span (creates hierarchy)
    name: str  # Operation name (e.g., "classify_request", "run_python_code")
    start_time: int  # When the operation started (unix ms)
    end_time: Optional[int] = None  # When it completed (unix ms)
    status: str = "STARTED"  # STARTED, COMPLETED, FAILED
    service: str  # Which service performed the operation
    module: str  # Which module/file contains the code
    function: str  # Which function was executing
    tags: Dict[str, str] = Field(default_factory=dict)  # Additional metadata
    metrics: Dict[str, float] = Field(default_factory=dict)  # Performance data

class TraceData(BaseModel):
    """
    Represents a full request/response cycle (e.g., processing a user query).
    Contains multiple spans showing the execution path.
    """
    trace_id: str  # Unique identifier for the trace
    user_prompt: str  # The user's original request/prompt
    start_time: int  # When processing started (unix ms)
    end_time: Optional[int] = None  # When processing completed (unix ms)
    status: str = "STARTED"  # STARTED, COMPLETED, FAILED
    tags: Dict[str, str] = Field(default_factory=dict)  # Additional metadata

class LogData(BaseModel):
    """
    Represents a log message with context about where it came from.
    """
    timestamp: int  # When the log was created (unix ms)
    level: str  # DEBUG, INFO, WARNING, ERROR
    message: str  # The log message
    trace_id: Optional[str] = None  # Which trace this log belongs to (if any)
    span_id: Optional[str] = None  # Which span this log belongs to (if any)
    service: str  # Which service created the log
    module: str  # Which module/file created the log
    tags: Dict[str, str] = Field(default_factory=dict)  # Additional metadata

class VariableData(BaseModel):
    """
    Captures the state of a variable during execution for debugging.
    """
    span_id: str  # Which span this variable belongs to
    trace_id: str  # Which trace this variable belongs to
    timestamp: int  # When the variable was captured (unix ms)
    service: str  # Which service captured the variable
    module: str  # Which module/file contains the variable
    function: str  # Which function contains the variable
    variable_name: str  # The variable's name
    variable_type: str  # The variable's type (str, int, dict, etc.)
    variable_value: Optional[str] = None  # String representation (may be omitted for privacy)
    tags: Dict[str, str] = Field(default_factory=dict)  # Additional metadata

class BatchData(BaseModel):
    """
    Batch submission of multiple data types to reduce API calls.
    """
    metrics: List[MetricData] = Field(default_factory=list)
    spans: List[SpanData] = Field(default_factory=list)
    traces: List[TraceData] = Field(default_factory=list)
    logs: List[LogData] = Field(default_factory=list)
    variables: List[VariableData] = Field(default_factory=list)

# Helper functions
def create_trace_id() -> str:
    """Generate a unique trace ID."""
    return str(uuid.uuid4())

def create_span_id() -> str:
    """Generate a unique span ID."""
    return str(uuid.uuid4())

def current_milli_time() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)

# API endpoints
@app.post("/api/v1/metric", status_code=201)
async def add_metric(metric: MetricData):
    """
    Add a single metric to InfluxDB.
    
    Metrics are stored in InfluxDB where they can be efficiently queried
    by time ranges and aggregated for visualization.
    """
    try:
        # If no timestamp provided, use current time
        timestamp = metric.timestamp or current_milli_time()
        
        # Create InfluxDB point - this is the data structure that InfluxDB expects
        point = Point(metric.name)
        
        # Add tags - these are indexed and can be used for filtering
        for tag_key, tag_value in metric.tags.items():
            point = point.tag(tag_key, tag_value)
        
        # Add fields and time - fields are the actual values being stored
        point = point.field("value", metric.value)
        point = point.time(timestamp)
        
        # Write to InfluxDB
        influx_write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        
        logger.debug(f"Added metric: {metric.name} = {metric.value}")
        return {"status": "success", "metric": metric.name}
    
    except Exception as e:
        logger.error(f"Error adding metric: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding metric: {str(e)}")

@app.post("/api/v1/span", status_code=201)
async def add_span(span: SpanData):
    """
    Add a span to MongoDB.
    
    Spans represent individual operations within a trace and are stored
    in MongoDB where their hierarchical structure can be easily preserved.
    """
    try:
        # Convert to dictionary for MongoDB
        span_dict = span.dict()
        
        # Ensure timestamps
        if not span_dict.get("end_time") and span_dict.get("status") == "COMPLETED":
            span_dict["end_time"] = current_milli_time()
        
        # Store in MongoDB
        await spans_collection.insert_one(span_dict)
        
        # Add basic metrics to InfluxDB - duration is particularly important
        # so we can easily track performance over time
        if span_dict.get("end_time"):
            duration = span_dict["end_time"] - span_dict["start_time"]
            duration_point = Point("span_duration")\
                .tag("span_id", span_dict["span_id"])\
                .tag("trace_id", span_dict["trace_id"])\
                .tag("name", span_dict["name"])\
                .tag("service", span_dict["service"])\
                .field("duration_ms", duration)\
                .time(span_dict["end_time"])
            
            influx_write_api.write(bucket=INFLUXDB_BUCKET, record=duration_point)
        
        # Add custom metrics from span - these can be any numeric values
        # the developer wants to track
        for metric_name, metric_value in span_dict.get("metrics", {}).items():
            metric_point = Point(f"span_metric_{metric_name}")\
                .tag("span_id", span_dict["span_id"])\
                .tag("trace_id", span_dict["trace_id"])\
                .tag("name", span_dict["name"])\
                .tag("service", span_dict["service"])\
                .field("value", metric_value)\
                .time(span_dict.get("end_time") or span_dict["start_time"])
            
            influx_write_api.write(bucket=INFLUXDB_BUCKET, record=metric_point)
        
        logger.debug(f"Added span: {span_dict['name']} (ID: {span_dict['span_id']})")
        return {"status": "success", "span_id": span_dict["span_id"]}
    
    except Exception as e:
        logger.error(f"Error adding span: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding span: {str(e)}")

@app.post("/api/v1/trace", status_code=201)
async def add_trace(trace: TraceData):
    """
    Add a trace to MongoDB.
    
    Traces represent entire request/response cycles and are the top level
    of the tracing hierarchy. Each trace contains multiple spans.
    """
    try:
        # Convert to dictionary for MongoDB
        trace_dict = trace.dict()
        
        # Ensure timestamps
        if not trace_dict.get("end_time") and trace_dict.get("status") == "COMPLETED":
            trace_dict["end_time"] = current_milli_time()
        
        # Store in MongoDB
        await traces_collection.insert_one(trace_dict)
        
        # Add basic metrics to InfluxDB if completed
        if trace_dict.get("end_time"):
            duration = trace_dict["end_time"] - trace_dict["start_time"]
            duration_point = Point("trace_duration")\
                .tag("trace_id", trace_dict["trace_id"])\
                .field("duration_ms", duration)\
                .time(trace_dict["end_time"])
            
            influx_write_api.write(bucket=INFLUXDB_BUCKET, record=duration_point)
        
        logger.debug(f"Added trace: {trace_dict['trace_id']}")
        return {"status": "success", "trace_id": trace_dict["trace_id"]}
    
    except Exception as e:
        logger.error(f"Error adding trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding trace: {str(e)}")

@app.post("/api/v1/log", status_code=201)
async def add_log(log: LogData):
    """
    Add a log entry to MongoDB.
    
    Logs provide context and information about what happened during execution.
    They are stored in MongoDB and linked to traces and spans when possible.
    """
    try:
        # Convert to dictionary for MongoDB
        log_dict = log.dict()
        
        # Add timestamp if not present
        if not log_dict.get("timestamp"):
            log_dict["timestamp"] = current_milli_time()
        
        # Store in MongoDB
        await logs_collection.insert_one(log_dict)
        
        # Add count metric to InfluxDB - this allows us to track log
        # volume and error rates over time
        log_point = Point("log_count")\
            .tag("level", log_dict["level"])\
            .tag("service", log_dict["service"])\
            .tag("module", log_dict["module"])
        
        if log_dict.get("trace_id"):
            log_point = log_point.tag("trace_id", log_dict["trace_id"])
        if log_dict.get("span_id"):
            log_point = log_point.tag("span_id", log_dict["span_id"])
        
        log_point = log_point.field("count", 1).time(log_dict["timestamp"])
        influx_write_api.write(bucket=INFLUXDB_BUCKET, record=log_point)
        
        logger.debug(f"Added log: {log_dict['level']} from {log_dict['service']}")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error adding log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding log: {str(e)}")

@app.post("/api/v1/variable", status_code=201)
async def add_variable(variable: VariableData):
    """
    Add a variable debugging information to MongoDB.
    
    Variables capture the state of important data during execution
    for debugging and understanding the system's behavior.
    """
    try:
        # Convert to dictionary for MongoDB
        var_dict = variable.dict()
        
        # Store in MongoDB
        await variables_collection.insert_one(var_dict)
        
        logger.debug(f"Added variable: {var_dict['variable_name']} from {var_dict['function']}")
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error adding variable: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding variable: {str(e)}")

@app.post("/api/v1/batch", status_code=201)
async def add_batch(batch: BatchData):
    """
    Add multiple metrics, spans, traces, logs, and variables in a single batch.
    
    Batching reduces the number of HTTP requests and improves performance
    when instrumenting code with many metric points.
    """
    try:
        # Process metrics
        for metric in batch.metrics:
            await add_metric(metric)
        
        # Process spans
        for span in batch.spans:
            await add_span(span)
        
        # Process traces
        for trace in batch.traces:
            await add_trace(trace)
        
        # Process logs
        for log in batch.logs:
            await add_log(log)
        
        # Process variables
        for variable in batch.variables:
            await add_variable(variable)
        
        return {
            "status": "success",
            "processed": {
                "metrics": len(batch.metrics),
                "spans": len(batch.spans),
                "traces": len(batch.traces),
                "logs": len(batch.logs),
                "variables": len(batch.variables)
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring system uptime."""
    return {"status": "ok", "timestamp": current_milli_time()}

# Main entry point - start the FastAPI server if this file is run directly
if __name__ == "__main__":
    uvicorn.run("collector:app", host="0.0.0.0", port=5001, reload=True)

Now let's create the API Gateway that will provide endpoints for querying metrics and traces:

#!/usr/bin/env python3
"""
Metrics API Gateway
------------------
Provides REST endpoints for querying metrics, spans, traces and logs
from the LLM Agent system.

This API acts as an abstraction layer between the frontend dashboard
and the underlying databases (InfluxDB and MongoDB), providing:
1. Query capabilities for time-series metrics
2. Trace and span retrieval with rich filtering
3. Log search and aggregation
4. Flame graph generation for visualizing execution paths
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import uvicorn
import requests
from fastapi import FastAPI, HTTPException, Query, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# InfluxDB for metrics queries - we use InfluxDB's Flux query language
# to perform time-series analysis and aggregations
from influxdb_client import InfluxDBClient
from influxdb_client.client.query_api import QueryApi

# MongoDB for traces and spans queries - MongoDB's flexible query language
# lets us efficiently filter and retrieve trace and span documents
import motor.motor_asyncio
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("metrics-api")

# Initialize FastAPI app - provides our HTTP endpoints with automatic
# OpenAPI documentation and request validation
app = FastAPI(title="LLM Agent Metrics API")

# Add CORS middleware to allow cross-origin requests from the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables - makes the service configurable
# without code changes and supports different deployment environments
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN", "your-super-secret-influx-token")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "llm_agent_org")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "metrics")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "agent_traces")

COLLECTOR_URL = os.getenv("COLLECTOR_URL", "http://localhost:5001")

# Initialize InfluxDB client - for querying time-series metrics
influxdb_client = InfluxDBClient(
    url=INFLUXDB_URL, 
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG
)
influx_query_api = influxdb_client.query_api()

# Initialize MongoDB client - for querying trace data and logs
mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
mongodb_database = mongodb_client[MONGODB_DB]
spans_collection = mongodb_database["spans"]
traces_collection = mongodb_database["traces"]
logs_collection = mongodb_database["logs"]
variables_collection = mongodb_database["variables"]

# Pydantic models for request/response validation
# These define the data structures for our API and enforce type checking

class TimeRange(BaseModel):
    """Time range for queries with optional start and end times."""
    start: Optional[int] = None  # Unix timestamp in milliseconds
    end: Optional[int] = None    # Unix timestamp in milliseconds

class MetricQuery(BaseModel):
    """
    Query parameters for time-series metrics.
    This structure allows flexible querying of metrics with various
    aggregations and filters.
    """
    name: str  # The metric name to query
    aggregation: str = "mean"  # Aggregation function: mean, sum, count, min, max
    interval: str = "1m"       # Time bucket size: 1s, 1m, 5m, 1h, etc.
    filters: Dict[str, str] = Field(default_factory=dict)  # Tag filters
    time_range: TimeRange = Field(default_factory=TimeRange)  # Query time window

class MetricDataPoint(BaseModel):
    """A single data point in a time-series metric."""
    timestamp: int  # When the measurement was taken
    value: float    # The measured value

class MetricResponse(BaseModel):
    """Response for a metric query with metadata and data points."""
    name: str             # The metric name
    aggregation: str      # The aggregation function used
    interval: str         # The time bucket size used
    data_points: List[MetricDataPoint]  # The actual time-series data

class TraceFilter(BaseModel):
    """
    Filter parameters for querying traces.
    This allows searching for traces by various criteria.
    """
    trace_id: Optional[str] = None
    user_prompt_contains: Optional[str] = None
    status: Optional[str] = None
    min_start_time: Optional[int] = None
    max_start_time: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class TraceResponse(BaseModel):
    """Response model for a trace query."""
    trace_id: str
    user_prompt: str
    start_time: int
    end_time: Optional[int] = None
    status: str
    duration_ms: Optional[int] = None
    tags: Dict[str, str]

class SpanFilter(BaseModel):
    """
    Filter parameters for querying spans.
    This allows searching for spans by various criteria.
    """
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    name_contains: Optional[str] = None
    service: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    status: Optional[str] = None
    min_start_time: Optional[int] = None
    max_start_time: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class SpanResponse(BaseModel):
    """Response model for a span query."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time: int
    end_time: Optional[int] = None
    status: str
    service: str
    module: str
    function: str
    duration_ms: Optional[int] = None
    tags: Dict[str, str]
    metrics: Dict[str, float]

class LogFilter(BaseModel):
    """
    Filter parameters for querying logs.
    This allows searching for logs by various criteria.
    """
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    level: Optional[str] = None
    message_contains: Optional[str] = None
    service: Optional[str] = None
    module: Optional[str] = None
    min_timestamp: Optional[int] = None
    max_timestamp: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class LogResponse(BaseModel):
    """Response model for a log query."""
    id: str
    timestamp: int
    level: str
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    service: str
    module: str
    tags: Dict[str, str]

class VariableFilter(BaseModel):
    """
    Filter parameters for querying variables.
    This allows searching for variables by various criteria.
    """
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    service: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    variable_name: Optional[str] = None
    variable_type: Optional[str] = None
    min_timestamp: Optional[int] = None
    max_timestamp: Optional[int] = None
    tags: Dict[str, str] = Field(default_factory=dict)

class VariableResponse(BaseModel):
    """Response model for a variable query."""
    id: str
    span_id: str
    trace_id: str
    timestamp: int
    service: str
    module: str
    function: str
    variable_name: str
    variable_type: str
    variable_value: Optional[str] = None
    tags: Dict[str, str]

# Flame graph models for visualizing execution hierarchies
class FlameGraphNode(BaseModel):
    """
    Node in a flame graph hierarchical visualization.
    Represents a single operation with its execution time and children.
    """
    id: str
    name: str
    value: int  # Duration in milliseconds
    color: Optional[str] = None
    children: List["FlameGraphNode"] = Field(default_factory=list)

# Forward reference for recursive structure
FlameGraphNode.update_forward_refs()

class FlameGraphResponse(BaseModel):
    """Response model for a flame graph query."""
    trace_id: str
    root: FlameGraphNode

# Helper functions
def current_milli_time() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)

def get_time_range_defaults(time_range: TimeRange) -> tuple:
    """
    Get default time range if not provided.
    Defaults to the last 24 hours if no start time is specified.
    """
    end_time = time_range.end or current_milli_time()
    # Default to last 24 hours if start not provided
    start_time = time_range.start or (end_time - 24 * 60 * 60 * 1000)
    return start_time, end_time

# API endpoints
@app.get("/api/v1/metrics", response_model=List[str])
async def list_metrics(prefix: str = Query(None)):
    """
    List available metric names, optionally filtered by prefix.
    
    This endpoint helps dashboard users discover what metrics are available
    for querying and visualization.
    """
    try:
        # Use InfluxDB's schema exploration capabilities to list all metrics
        query = '''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "%s")
        ''' % INFLUXDB_BUCKET
        
        tables = influx_query_api.query(query=query)
        
        metrics = []
        for table in tables:
            for record in table.records:
                metric_name = record.values.get("_value")
                if prefix is None or metric_name.startswith(prefix):
                    metrics.append(metric_name)
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error listing metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing metrics: {str(e)}")

@app.post("/api/v1/metrics/query", response_model=MetricResponse)
async def query_metrics(query: MetricQuery):
    """
    Query time-series metrics data with filters and aggregation.
    
    This endpoint is the core query interface for metrics dashboards,
    allowing flexible time-series analysis with various aggregations.
    """
    try:
        start_time, end_time = get_time_range_defaults(query.time_range)
        
        # Build Flux query with the specified filters and aggregation
        filters = []
        for key, value in query.filters.items():
            filters.append(f'r["{key}"] == "{value}"')
        
        filter_clause = " and ".join(filters) if filters else "true"
        
        # This Flux query retrieves time-series data with the specified
        # aggregation and time bucket size
        flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: {start_time}, stop: {end_time})
            |> filter(fn: (r) => r._measurement == "{query.name}")
            |> filter(fn: (r) => {filter_clause})
            |> aggregateWindow(every: {query.interval}, fn: {query.aggregation}, createEmpty: false)
            |> yield(name: "result")
        '''
        
        tables = influx_query_api.query(query=flux_query)
        
        data_points = []
        for table in tables:
            for record in table.records:
                timestamp = int(record.get_time().timestamp() * 1000)
                value = record.get_value()
                data_points.append(MetricDataPoint(timestamp=timestamp, value=value))
        
        return MetricResponse(
            name=query.name,
            aggregation=query.aggregation,
            interval=query.interval,
            data_points=data_points
        )
    
    except Exception as e:
        logger.error(f"Error querying metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying metrics: {str(e)}")

@app.get("/api/v1/traces", response_model=List[TraceResponse])
async def list_traces(
    trace_id: str = Query(None),
    user_prompt_contains: str = Query(None),
    status: str = Query(None),
    min_start_time: int = Query(None),
    max_start_time: int = Query(None),
    limit: int = Query(100, le=1000),
    skip: int = Query(0)
):
    """
    List traces with optional filtering.
    
    This endpoint lets clients search for traces based on various criteria
    like time range, status, or content of the user prompt.
    """
    try:
        # Build MongoDB query based on the provided filters
        query = {}
        if trace_id:
            query["trace_id"] = trace_id
        if user_prompt_contains:
            query["user_prompt"] = {"$regex": user_prompt_contains, "$options": "i"}
        if status:
            query["status"] = status
        
        # Time range filter
        time_query = {}
        if min_start_time:
            time_query["$gte"] = min_start_time
        if max_start_time:
            time_query["$lte"] = max_start_time
        if time_query:
            query["start_time"] = time_query
        
        # Execute query with pagination
        cursor = traces_collection.find(query).sort("start_time", -1).skip(skip).limit(limit)
        traces = []
        
        async for doc in cursor:
            # Calculate duration if possible
            duration_ms = None
            if doc.get("end_time") and doc.get("start_time"):
                duration_ms = doc["end_time"] - doc["start_time"]
            
            # Remove MongoDB _id (not serializable) and add our own id field
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Add duration
            doc["duration_ms"] = duration_ms
            
            traces.append(TraceResponse(**doc))
        
        return traces
    
    except Exception as e:
        logger.error(f"Error listing traces: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing traces: {str(e)}")

@app.get("/api/v1/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str = Path(...)):
    """
    Get a specific trace by ID.
    
    This endpoint retrieves detailed information about a single trace.
    """
    try:
        # Find the trace by its ID
        doc = await traces_collection.find_one({"trace_id": trace_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
        
        # Calculate duration if possible
        duration_ms = None
        if doc.get("end_time") and doc.get("start_time"):
            duration_ms = doc["end_time"] - doc["start_time"]
        
        # Remove MongoDB _id (not serializable) and add our own id field
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        
        # Add duration
        doc["duration_ms"] = duration_ms
        
        return TraceResponse(**doc)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trace: {str(e)}")

@app.get("/api/v1/spans", response_model=List[SpanResponse])
async def list_spans(
    span_id: str = Query(None),
    trace_id: str = Query(None),
    parent_span_id: str = Query(None),
    name_contains: str = Query(None),
    service: str = Query(None),
    module: str = Query(None),
    function: str = Query(None),
    status: str = Query(None),
    min_start_time: int = Query(None),
    max_start_time: int = Query(None),
    limit: int = Query(100, le=1000),
    skip: int = Query(0)
):
    """
    List spans with optional filtering.
    
    This endpoint lets clients search for spans based on various criteria
    like trace ID, service name, or function name.
    """
    try:
        # Build MongoDB query based on the provided filters
        query = {}
        if span_id:
            query["span_id"] = span_id
        if trace_id:
            query["trace_id"] = trace_id
        if parent_span_id:
            query["parent_span_id"] = parent_span_id
        if name_contains:
            query["name"] = {"$regex": name_contains, "$options": "i"}
        if service:
            query["service"] = service
        if module:
            query["module"] = module
        if function:
            query["function"] = function
        if status:
            query["status"] = status
        
        # Time range filter
        time_query = {}
        if min_start_time:
            time_query["$gte"] = min_start_time
        if max_start_time:
            time_query["$lte"] = max_start_time
        if time_query:
            query["start_time"] = time_query
        
        # Execute query with pagination
        cursor = spans_collection.find(query).sort("start_time", -1).skip(skip).limit(limit)
        spans = []
        
        async for doc in cursor:
            # Calculate duration if possible
            duration_ms = None
            if doc.get("end_time") and doc.get("start_time"):
                duration_ms = doc["end_time"] - doc["start_time"]
            
            # Remove MongoDB _id and add our own id field
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Add duration
            doc["duration_ms"] = duration_ms
            
            # Ensure metrics field exists
            if "metrics" not in doc:
                doc["metrics"] = {}
            
            spans.append(SpanResponse(**doc))
        
        return spans
    
    except Exception as e:
        logger.error(f"Error listing spans: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing spans: {str(e)}")

@app.get("/api/v1/spans/{span_id}", response_model=SpanResponse)
async def get_span(span_id: str = Path(...)):
    """
    Get a specific span by ID.
    
    This endpoint retrieves detailed information about a single span.
    """
    try:
        # Find the span by its ID
        doc = await spans_collection.find_one({"span_id": span_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Span not found: {span_id}")
        
        # Calculate duration if possible
        duration_ms = None
        if doc.get("end_time") and doc.get("start_time"):
            duration_ms = doc["end_time"] - doc["start_time"]
        
        # Remove MongoDB _id and add our own id field
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        
        # Add duration
        doc["duration_ms"] = duration_ms
        
        # Ensure metrics field exists
        if "metrics" not in doc:
            doc["metrics"] = {}
        
        return SpanResponse(**doc)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting span: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting span: {str(e)}")

@app.get("/api/v1/logs", response_model=List[LogResponse])
async def list_logs(
    trace_id: str = Query(None),
    span_id: str = Query(None),
    level: str = Query(None),
    message_contains: str = Query(None),
    service: str = Query(None),
    module: str = Query(None),
    min_timestamp: int = Query(None),
    max_timestamp: int = Query(None),
    limit: int = Query(100, le=1000),
    skip: int = Query(0)
):
    """
    List logs with optional filtering.
    
    This endpoint lets clients search for logs based on various criteria
    like trace ID, log level, or message content.
    """
    try:
        # Build MongoDB query based on the provided filters
        query = {}
        if trace_id:
            query["trace_id"] = trace_id
        if span_id:
            query["span_id"] = span_id
        if level:
            query["level"] = level
        if message_contains:
            query["message"] = {"$regex": message_contains, "$options": "i"}
        if service:
            query["service"] = service
        if module:
            query["module"] = module
        
        # Time range filter
        time_query = {}
        if min_timestamp:
            time_query["$gte"] = min_timestamp
        if max_timestamp:
            time_query["$lte"] = max_timestamp
        if time_query:
            query["timestamp"] = time_query
        
        # Execute query with pagination
        cursor = logs_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = []
        
        async for doc in cursor:
            # Remove MongoDB _id and add our own id field
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Ensure tags field exists
            if "tags" not in doc:
                doc["tags"] = {}
            
            logs.append(LogResponse(**doc))
        
        return logs
    
    except Exception as e:
        logger.error(f"Error listing logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing logs: {str(e)}")

@app.get("/api/v1/variables", response_model=List[VariableResponse])
async def list_variables(
    trace_id: str = Query(None),
    span_id: str = Query(None),
    service: str = Query(None),
    module: str = Query(None),
    function: str = Query(None),
    variable_name: str = Query(None),
    variable_type: str = Query(None),
    min_timestamp: int = Query(None),
    max_timestamp: int = Query(None),
    limit: int = Query(100, le=1000),
    skip: int = Query(0)
):
    """
    List variables with optional filtering.
    
    This endpoint lets clients search for captured variable values
    based on various criteria like trace ID, function, or variable name.
    """
    try:
        # Build MongoDB query based on the provided filters
        query = {}
        if trace_id:
            query["trace_id"] = trace_id
        if span_id:
            query["span_id"] = span_id
        if service:
            query["service"] = service
        if module:
            query["module"] = module
        if function:
            query["function"] = function
        if variable_name:
            query["variable_name"] = variable_name
        if variable_type:
            query["variable_type"] = variable_type
        
        # Time range filter
        time_query = {}
        if min_timestamp:
            time_query["$gte"] = min_timestamp
        if max_timestamp:
            time_query["$lte"] = max_timestamp
        if time_query:
            query["timestamp"] = time_query
        
        # Execute query with pagination
        cursor = variables_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        variables = []
        
        async for doc in cursor:
            # Remove MongoDB _id and add our own id field
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            
            # Ensure tags field exists
            if "tags" not in doc:
                doc["tags"] = {}
            
            variables.append(VariableResponse(**doc))
        
        return variables
    
    except Exception as e:
        logger.error(f"Error listing variables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing variables: {str(e)}")

@app.get("/api/v1/flame-graph/{trace_id}", response_model=FlameGraphResponse)
async def get_flame_graph(trace_id: str = Path(...)):
    """
    Generate a flame graph for a specific trace.
    
    Flame graphs visualize the hierarchical execution structure of a trace,
    showing how time is spent across different operations.
    """
    try:
        # First check if trace exists
        trace = await traces_collection.find_one({"trace_id": trace_id})
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
        
        # Find all spans for this trace
        spans = []
        async for span in spans_collection.find({"trace_id": trace_id}):
            # Calculate duration
            duration_ms = 0
            if span.get("end_time") and span.get("start_time"):
                duration_ms = span["end_time"] - span["start_time"]
            
            spans.append({
                "span_id": span["span_id"],
                "parent_span_id": span.get("parent_span_id"),
                "name": span["name"],
                "duration_ms": duration_ms,
                "service": span["service"],
                "module": span["module"],
                "function": span["function"]
            })
        
        if not spans:
            raise HTTPException(status_code=404, detail=f"No spans found for trace: {trace_id}")
        
        # Build the flame graph
        # First, organize spans in a hierarchy
        span_map = {span["span_id"]: span for span in spans}
        child_map = {}
        
        # Map parent-child relationships
        for span in spans:
            parent_id = span.get("parent_span_id")
            if parent_id:
                if parent_id not in child_map:
                    child_map[parent_id] = []
                child_map[parent_id].append(span["span_id"])
        
        # Find the root span (no parent or parent doesn't exist)
        root_spans = []
        for span in spans:
            parent_id = span.get("parent_span_id")
            if not parent_id or parent_id not in span_map:
                root_spans.append(span)
        
        if not root_spans:
            # If no clear root, just use the first span
            root_spans = [spans[0]]
        
        # Generate color based on service/module
        color_map = {}
        services = set(span["service"] for span in spans)
        functions = set(span["function"] for span in spans)
        
        # Simple color scheme based on services
        colors = [
            "#4a82bd", "#c0504d", "#9bbb59", "#8064a2", "#4bacc6", 
            "#f79646", "#7f7f7f", "#ffff00", "#00b050", "#7030a0"
        ]
        
        for idx, service in enumerate(services):
            color_map[service] = colors[idx % len(colors)]
        
        # Build the flame graph recursively
        def build_node(span_id):
            span = span_map[span_id]
            children = []
            
            if span_id in child_map:
                for child_id in child_map[span_id]:
                    children.append(build_node(child_id))
            
            service = span["service"]
            
            return FlameGraphNode(
                id=span_id,
                name=f"{span['name']} ({span['function']})",
                value=max(span["duration_ms"], 1),  # Ensure at least 1ms duration for visibility
                color=color_map.get(service),
                children=children
            )
        
        # Build the root node
        root = FlameGraphNode(
            id="root",
            name=f"Trace: {trace_id}",
            value=trace.get("end_time", 0) - trace.get("start_time", 0),
            color="#333333",
            children=[build_node(root_span["span_id"]) for root_span in root_spans]
        )
        
        return FlameGraphResponse(
            trace_id=trace_id,
            root=root
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating flame graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating flame graph: {str(e)}")

@app.get("/api/v1/services", response_model=List[str])
async def list_services():
    """
    List all services that have submitted spans.
    
    This helps dashboard users understand what services are being monitored.
    """
    try:
        services = await spans_collection.distinct("service")
        return services
    
    except Exception as e:
        logger.error(f"Error listing services: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing services: {str(e)}")

@app.get("/api/v1/modules", response_model=List[str])
async def list_modules(service: str = Query(None)):
    """
    List all modules, optionally filtered by service.
    
    This helps dashboard users understand what modules exist within services.
    """
    try:
        query = {}
        if service:
            query["service"] = service
        
        modules = await spans_collection.distinct("module", query)
        return modules
    
    except Exception as e:
        logger.error(f"Error listing modules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing modules: {str(e)}")

@app.get("/api/v1/functions", response_model=List[str])
async def list_functions(service: str = Query(None), module: str = Query(None)):
    """
    List all functions, optionally filtered by service and module.
    
    This helps dashboard users understand what functions exist within modules.
    """
    try:
        query = {}
        if service:
            query["service"] = service
        if module:
            query["module"] = module
        
        functions = await spans_collection.distinct("function", query)
        return functions
    
    except Exception as e:
        logger.error(f"Error listing functions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing functions: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    This endpoint lets monitoring systems verify that the API is running
    and can communicate with the collector service.
    """
    # Also check collector health
    try:
        collector_response = requests.get(f"{COLLECTOR_URL}/health", timeout=5)
        collector_status = "ok" if collector_response.status_code == 200 else "error"
    except Exception as e:
        collector_status = f"error: {str(e)}"
    
    return {
        "status": "ok", 
        "timestamp": current_milli_time(),
        "collector_status": collector_status
    }

# Main entry point - start the FastAPI server if this file is run directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

Now, let's create the instrumentation client library that can be used in the LLM agent to send metrics to our collection service:

"""
Metrics Client for LLM Agent
---------------------------
Client library for easy instrumentation of the LLM agent code.
Provides decorators and context managers for capturing spans, logs, and metrics.

This client library makes it simple to add observability to your LLM agent
without cluttering your business logic with monitoring code. It provides:

1. Automatic trace and span creation with parent-child relationships
2. Function timing with decorators
3. Variable state capture for debugging
4. Log forwarding to the central monitoring system
5. Batched sending of metrics for performance
"""

import os
import sys
import time
import uuid
import json
import inspect
import logging
import threading
import functools
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import contextmanager

# Requests library for communicating with the metrics collector service
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("metrics-client")

# Default settings - can be overridden via environment variables
# or the configure_metrics() function
DEFAULT_COLLECTOR_URL = os.getenv("METRICS_COLLECTOR_URL", "http://localhost:5001")
DEFAULT_SERVICE_NAME = os.getenv("METRICS_SERVICE_NAME", "llm_agent")
DEFAULT_BATCH_SIZE = int(os.getenv("METRICS_BATCH_SIZE", "10"))
DEFAULT_FLUSH_INTERVAL = int(os.getenv("METRICS_FLUSH_INTERVAL", "5"))  # seconds

class MetricsClientConfig:
    """
    Configuration for the MetricsClient.
    
    This class holds settings for the metrics client, including:
    - collector_url: The URL of the metrics collector service
    - service_name: The name of this service (used in spans)
    - batch_size: Number of metrics to accumulate before sending
    - flush_interval: Maximum time to wait before sending metrics
    - disable: Flag to completely disable metrics collection
    """
    
    def __init__(
        self,
        collector_url: str = DEFAULT_COLLECTOR_URL,
        service_name: str = DEFAULT_SERVICE_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: int = DEFAULT_FLUSH_INTERVAL,
        disable: bool = False
    ):
        self.collector_url = collector_url
        self.service_name = service_name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.disable = disable

class MetricsClient:
    """
    Client for sending metrics, spans, and logs to the collector service.
    
    This is implemented as a singleton class, so all parts of the application
    share the same client instance and can contribute to the same traces.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config: Optional[MetricsClientConfig] = None) -> 'MetricsClient':
        """
        Get or create the singleton instance of MetricsClient.
        
        This ensures all code is working with the same client instance
        for proper trace coordination.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config or MetricsClientConfig())
        return cls._instance
    
    def __init__(self, config: MetricsClientConfig):
        """
        Initialize the client with the given configuration.
        
        This constructor is not meant to be called directly.
        Use get_instance() instead.
        """
        self.config = config
        
        # Batches for metrics, spans, traces, logs, and variables
        # These will be flushed periodically or when they reach batch_size
        self.metrics_batch = []
        self.spans_batch = []
        self.traces_batch = []
        self.logs_batch = []
        self.variables_batch = []
        
        # Currently active trace and spans
        # These allow us to associate related operations
        self.active_trace_id = None
        self.active_spans = {}  # span_id -> span_data
        
        # For background flushing
        self.last_flush_time = time.time()
        self.batch_lock = threading.Lock()
        
        # Configure request session with retry logic for robustness
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retry))
        self.session.mount('https://', HTTPAdapter(max_retries=retry))
        
        # Start background flush thread if batching is enabled
        if not self.config.disable and self.config.flush_interval > 0:
            self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
            self.flush_thread.start()
    
    def _background_flush(self):
        """
        Background thread to periodically flush metrics.
        
        This ensures metrics are sent even if batch_size isn't reached,
        so we don't lose data if activity is low.
        """
        while True:
            time.sleep(1)  # Check every second
            
            current_time = time.time()
            if current_time - self.last_flush_time >= self.config.flush_interval:
                self.flush()
    
    def flush(self):
        """
        Flush all pending metrics, spans, and logs to the collector.
        
        This sends all batched data to the collector service and
        clears the local batches.
        """
        if self.config.disable:
            return
        
        with self.batch_lock:
            # Skip if no data to send
            if not (self.metrics_batch or self.spans_batch or self.traces_batch or 
                    self.logs_batch or self.variables_batch):
                self.last_flush_time = time.time()
                return
            
            # Copy batches for sending
            metrics = self.metrics_batch.copy()
            spans = self.spans_batch.copy()
            traces = self.traces_batch.copy()
            logs = self.logs_batch.copy()
            variables = self.variables_batch.copy()
            
            # Clear batches
            self.metrics_batch = []
            self.spans_batch = []
            self.traces_batch = []
            self.logs_batch = []
            self.variables_batch = []
        
        try:
            # Prepare batch payload for the collector service
            payload = {
                "metrics": metrics,
                "spans": spans,
                "traces": traces,
                "logs": logs,
                "variables": variables
            }
            
            # Send to collector service
            response = self.session.post(
                f"{self.config.collector_url}/api/v1/batch",
                json=payload,
                timeout=10
            )
            
            if response.status_code != 201:
                logger.warning(f"Failed to flush metrics batch: {response.status_code} {response.text}")
            else:
                logger.debug(f"Flushed batch: {len(metrics)} metrics, {len(spans)} spans, "
                             f"{len(traces)} traces, {len(logs)} logs, {len(variables)} variables")
        
        except Exception as e:
            logger.error(f"Error flushing metrics batch: {str(e)}")
        
        self.last_flush_time = time.time()
    
    def add_metric(self, name: str, value: float, timestamp: Optional[int] = None,
                  tags: Optional[Dict[str, str]] = None):
        """
        Add a metric to the batch.
        
        Metrics are numeric values that change over time, like:
        - Function execution times
        - Memory usage
        - API call counts
        - Error rates
        """
        if self.config.disable:
            return
        
        # Create the metric data structure
        metric = {
            "name": name,
            "value": value,
            "timestamp": timestamp or int(time.time() * 1000),
            "tags": tags or {}
        }
        
        # Add trace ID if one is active
        if self.active_trace_id:
            metric["tags"]["trace_id"] = self.active_trace_id
        
        # Add to batch and flush if batch size reached
        with self.batch_lock:
            self.metrics_batch.append(metric)
            if len(self.metrics_batch) >= self.config.batch_size:
                self.flush()
    
    def start_span(self, name: str, parent_span_id: Optional[str] = None,
                  module: Optional[str] = None, function: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new span and return its ID.
        
        A span represents a single operation or unit of work in a trace, like:
        - A function call
        - An API request
        - A database query
        - A code execution step
        
        Spans form a hierarchy through parent-child relationships.
        """
        if self.config.disable:
            return str(uuid.uuid4())  # Return dummy ID when disabled
        
        # Auto-create a trace if none is active
        trace_id = self.active_trace_id
        if not trace_id:
            # Auto-create a trace if none is active
            trace_id = self.start_trace("auto_trace", {})
        
        # Get caller info if not provided (auto-discovery)
        if not module or not function:
            frame = inspect.currentframe().f_back
            if frame:
                module = module or frame.f_globals.get('__name__', 'unknown_module')
                function = function or frame.f_code.co_name
        
        # Create span with a unique ID
        span_id = str(uuid.uuid4())
        span = {
            "span_id": span_id,
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "name": name,
            "start_time": int(time.time() * 1000),
            "status": "STARTED",
            "service": self.config.service_name,
            "module": module or "unknown_module",
            "function": function or "unknown_function",
            "tags": tags or {},
            "metrics": {}
        }
        
        # Store span and add to batch
        with self.batch_lock:
            self.active_spans[span_id] = span
            self.spans_batch.append(span)
            if len(self.spans_batch) >= self.config.batch_size:
                self.flush()
        
        return span_id
    
    def end_span(self, span_id: str, status: str = "COMPLETED",
                metrics: Optional[Dict[str, float]] = None):
        """
        End a span with the given status and metrics.
        
        This marks a span as completed and records its end time.
        Optional metrics can be attached to provide additional context.
        """
        if self.config.disable or span_id not in self.active_spans:
            return
        
        end_time = int(time.time() * 1000)
        
        with self.batch_lock:
            # Get the span from active spans
            span = self.active_spans.pop(span_id, None)
            if not span:
                logger.warning(f"Attempted to end unknown span: {span_id}")
                return
            
            # Create an updated span with end info
            updated_span = span.copy()
            updated_span["end_time"] = end_time
            updated_span["status"] = status
            if metrics:
                updated_span["metrics"] = metrics
            
            # Add the updated span to the batch
            self.spans_batch.append(updated_span)
            if len(self.spans_batch) >= self.config.batch_size:
                self.flush()
    
    def start_trace(self, user_prompt: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new trace and set it as active. Returns the trace ID.
        
        A trace represents a full request/response cycle, like:
        - Processing a user query
        - Executing a command
        - Running a workflow
        
        Each trace contains multiple spans showing the execution path.
        """
        if self.config.disable:
            return str(uuid.uuid4())  # Return dummy ID when disabled
        
        # Create trace with a unique ID
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "user_prompt": user_prompt,
            "start_time": int(time.time() * 1000),
            "status": "STARTED",
            "tags": tags or {}
        }
        
        # Store trace and add to batch
        with self.batch_lock:
            self.active_trace_id = trace_id
            self.traces_batch.append(trace)
            if len(self.traces_batch) >= self.config.batch_size:
                self.flush()
        
        return trace_id
    
    def end_trace(self, trace_id: Optional[str] = None, status: str = "COMPLETED"):
        """
        End the active trace with the given status.
        
        This marks a trace as completed and records its end time.
        """
        if self.config.disable:
            return
        
        trace_id = trace_id or self.active_trace_id
        if not trace_id:
            logger.warning("Attempted to end trace when no trace is active")
            return
        
        end_time = int(time.time() * 1000)
        
        with self.batch_lock:
            # Create an updated trace with end info
            updated_trace = {
                "trace_id": trace_id,
                "end_time": end_time,
                "status": status
            }
            
            # Add the updated trace to the batch
            self.traces_batch.append(updated_trace)
            
            # Clear active trace ID if it matches the one being ended
            if self.active_trace_id == trace_id:
                self.active_trace_id = None
            
            if len(self.traces_batch) >= self.config.batch_size:
                self.flush()
    
    def log(self, level: str, message: str, span_id: Optional[str] = None,
           module: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Add a log entry to the batch.
        
        This lets us record events and messages during execution,
        linked to the relevant trace and span.
        """
        if self.config.disable:
            return
        
        # Get caller info if not provided (auto-discovery)
        if not module:
            frame = inspect.currentframe().f_back
            if frame:
                module = frame.f_globals.get('__name__', 'unknown_module')
        
        # Determine span and trace to associate with the log
        trace_id = self.active_trace_id
        if not span_id and self.active_spans:
            # Use the most recently started span if none specified
            latest_span_id = max(self.active_spans.items(),
                                key=lambda x: x[1]["start_time"])[0]
            span_id = latest_span_id
        
        # Create log entry
        log_entry = {
            "timestamp": int(time.time() * 1000),
            "level": level,
            "message": message,
            "service": self.config.service_name,
            "module": module or "unknown_module",
            "tags": tags or {}
        }
        
        # Associate with trace and span if available
        if trace_id:
            log_entry["trace_id"] = trace_id
        if span_id:
            log_entry["span_id"] = span_id
        
        # Add to batch and flush if needed
        with self.batch_lock:
            self.logs_batch.append(log_entry)
            if len(self.logs_batch) >= self.config.batch_size:
                self.flush()
    
    def capture_variable(self, variable_name: str, variable_value: Any, variable_type: Optional[str] = None,
                        span_id: Optional[str] = None, function: Optional[str] = None,
                        module: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Capture a variable's value for debugging.
        
        This lets us record the state of important variables during execution,
        which is invaluable for debugging complex issues.
        """
        if self.config.disable:
            return
        
        # Get caller info if not provided (auto-discovery)
        if not module or not function:
            frame = inspect.currentframe().f_back
            if frame:
                module = module or frame.f_globals.get('__name__', 'unknown_module')
                function = function or frame.f_code.co_name
        
        # Determine span and trace to associate with the variable
        trace_id = self.active_trace_id
        if not span_id and self.active_spans:
            # Use the most recently started span if none specified
            latest_span_id = max(self.active_spans.items(),
                                key=lambda x: x[1]["start_time"])[0]
            span_id = latest_span_id
        
        # Validate we have the required associations
        if not span_id:
            logger.warning(f"No active span for variable {variable_name}")
            return
        
        if not trace_id:
            logger.warning(f"No active trace for variable {variable_name}")
            return
        
        # Determine variable type if not provided
        if variable_type is None:
            if variable_value is None:
                variable_type = "None"
            else:
                variable_type = type(variable_value).__name__
        
        # Convert value to string if present
        # We limit the string length for privacy and performance
        value_str = None
        if variable_value is not None:
            try:
                if isinstance(variable_value, (str, int, float, bool)):
                    value_str = str(variable_value)
                else:
                    # For complex objects, limit to a short representation
                    value_str = str(variable_value)[:100]
            except Exception:
                value_str = "<Error converting to string>"
        
        # Create variable entry
        variable_entry = {
            "span_id": span_id,
            "trace_id": trace_id,
            "timestamp": int(time.time() * 1000),
            "service": self.config.service_name,
            "module": module or "unknown_module",
            "function": function or "unknown_function",
            "variable_name": variable_name,
            "variable_type": variable_type,
            "variable_value": value_str,
            "tags": tags or {}
        }
        
        # Add to batch and flush if needed
        with self.batch_lock:
            self.variables_batch.append(variable_entry)
            if len(self.variables_batch) >= self.config.batch_size:
                self.flush()

# Convenience functions using the singleton client
# These provide a simpler API for common operations

def configure_metrics(
    collector_url: Optional[str] = None,
    service_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    flush_interval: Optional[int] = None,
    disable: bool = False
):
    """
    Configure the metrics client singleton.
    
    This should be called early in your application's startup
    to set global configuration for metrics collection.
    """
    config = MetricsClientConfig(
        collector_url=collector_url or DEFAULT_COLLECTOR_URL,
        service_name=service_name or DEFAULT_SERVICE_NAME,
        batch_size=batch_size or DEFAULT_BATCH_SIZE,
        flush_interval=flush_interval or DEFAULT_FLUSH_INTERVAL,
        disable=disable
    )
    
    MetricsClient.get_instance(config)
    logger.info(f"Metrics client configured: service={config.service_name}, "
                f"collector={config.collector_url}, batching={not config.disable}")

def add_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Add a metric using the singleton client."""
    MetricsClient.get_instance().add_metric(name, value, tags=tags)

def log_debug(message: str, span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Log a debug message using the singleton client."""
    MetricsClient.get_instance().log("DEBUG", message, span_id=span_id, tags=tags)

def log_info(message: str, span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Log an info message using the singleton client."""
    MetricsClient.get_instance().log("INFO", message, span_id=span_id, tags=tags)

def log_warning(message: str, span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Log a warning message using the singleton client."""
    MetricsClient.get_instance().log("WARNING", message, span_id=span_id, tags=tags)

def log_error(message: str, span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Log an error message using the singleton client."""
    MetricsClient.get_instance().log("ERROR", message, span_id=span_id, tags=tags)

def capture_variable(name: str, value: Any, span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Capture a variable using the singleton client."""
    MetricsClient.get_instance().capture_variable(name, value, span_id=span_id, tags=tags)

def start_trace(user_prompt: str, tags: Optional[Dict[str, str]] = None) -> str:
    """Start a new trace using the singleton client."""
    return MetricsClient.get_instance().start_trace(user_prompt, tags=tags)

def end_trace(trace_id: Optional[str] = None, status: str = "COMPLETED"):
    """End the active trace using the singleton client."""
    MetricsClient.get_instance().end_trace(trace_id, status=status)

def start_span(name: str, parent_span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
    """Start a new span using the singleton client."""
    return MetricsClient.get_instance().start_span(name, parent_span_id=parent_span_id, tags=tags)

def end_span(span_id: str, status: str = "COMPLETED", metrics: Optional[Dict[str, float]] = None):
    """End a span using the singleton client."""
    MetricsClient.get_instance().end_span(span_id, status=status, metrics=metrics)

def flush():
    """Flush all pending metrics using the singleton client."""
    MetricsClient.get_instance().flush()

# Useful decorators and context managers
# These provide syntactic sugar for common instrumentation patterns

@contextmanager
def trace_context(user_prompt: str, tags: Optional[Dict[str, str]] = None):
    """
    Context manager for tracing a block of code.
    
    Usage:
    ```python
    with trace_context("Processing user query", {"priority": "high"}):
        # Code to process the query
        result = process_query(query)
    ```
    
    This automatically starts a trace, executes the code block,
    and ends the trace when the block exits.
    """
    trace_id = start_trace(user_prompt, tags=tags)
    try:
        yield trace_id
    except Exception as e:
        log_error(f"Error in trace context: {str(e)}")
        end_trace(trace_id, status="FAILED")
        raise
    else:
        end_trace(trace_id)

@contextmanager
def span_context(name: str, parent_span_id: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Context manager for spanning a block of code.
    
    Usage:
    ```python
    with span_context("database_query", tags={"table": "users"}):
        # Code to execute the database query
        results = db.query("SELECT * FROM users")
    ```
    
    This automatically starts a span, executes the code block,
    and ends the span when the block exits.
    """
    span_id = start_span(name, parent_span_id=parent_span_id, tags=tags)
    start_time = time.time()
    try:
        yield span_id
    except Exception as e:
        log_error(f"Error in span context: {str(e)}")
        duration_ms = int((time.time() - start_time) * 1000)
        end_span(span_id, status="FAILED", metrics={"duration_ms": duration_ms})
        raise
    else:
        duration_ms = int((time.time() - start_time) * 1000)
        end_span(span_id, metrics={"duration_ms": duration_ms})

def traced(tags: Optional[Dict[str, str]] = None):
    """
    Decorator to trace a function execution.
    
    Usage:
    ```python
    @traced(tags={"importance": "high"})
    def process_user_query(query):
        # Process the query
        return result
    ```
    
    This automatically creates a trace for the function call.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_context(f"{func.__name__}", tags=tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def spanned(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to span a function execution.
    
    Usage:
    ```python
    @spanned(tags={"module": "database"})
    def fetch_user(user_id):
        # Fetch the user
        return user
    ```
    
    This automatically creates a span for the function call
    with duration tracking.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with span_context(span_name, tags=tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def timed(metric_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to time a function execution and record it as a metric.
    
    Usage:
    ```python
    @timed(metric_name="api.response_time", tags={"endpoint": "/users"})
    def get_users():
        # Fetch users
        return users
    ```
    
    This measures the execution time of the function and
    records it as a metric.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                name = metric_name or f"function.duration.{func.__module__}.{func.__name__}"
                add_metric(name, duration_ms, tags=tags)
        return wrapper
    return decorator

# Initialize metrics with default settings when this module is imported
configure_metrics()

Finally, let's create a README.md file that will serve as an onboarding guide for new engineers:

# LLM Agent Metrics System

Welcome to the LLM Agent Metrics System! This guide will help you understand and use our monitoring and observability stack to gain insights into your LLM Agent's performance and behavior.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start Guide](#quick-start-guide)
4. [Technology Choices](#technology-choices)
5. [Core Concepts](#core-concepts)
6. [Implementation Guide](#implementation-guide)
7. [UX Endpoints & Documentation](#ux-endpoints--documentation)
8. [FAQ](#faq)
9. [Research Topics](#research-topics)
10. [Strategy Guide](#strategy-guide)

## Overview

The LLM Agent Metrics System is a comprehensive monitoring solution designed specifically for LLM Agents. It provides:

- **Performance Monitoring**: Track execution times, resource usage, and throughput
- **Distributed Tracing**: Follow requests as they flow through different components
- **Debug Information**: Capture variable states and execution paths for debugging
- **Log Centralization**: Collect and search logs from all services
- **Visualization**: Interactive dashboards for exploring metrics and traces

This system is designed to be lightweight, easy to integrate, and scalable to handle high throughput LLM Agent deployments.

## Architecture

The system consists of five main components:

1. **Metrics Client**: A Python library for instrumenting your code (metrics_client.py)
2. **Metrics Collector**: A service that receives and processes metrics data
3. **Time-Series Database**: InfluxDB for storing time-series metrics
4. **Document Database**: MongoDB for storing traces, spans, and logs
5. **Dashboard**: A React frontend for visualizing metrics and traces

Here's a high-level diagram of how these components interact:

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  LLM Agent     │     │  Metrics       │     │  API Gateway   │
│  (Your Code)   │────▶│  Collector     │◀────│                │
│  + Metrics     │     │  Service       │     │                │
│    Client      │     └────┬───────┬───┘     └────┬───────────┘
└────────────────┘          │       │              │
                           ▼       ▼              ▼
                   ┌─────────┐    ┌─────────┐    ┌─────────────┐
                   │InfluxDB │    │MongoDB  │    │ Dashboard   │
                   │(Metrics)│    │(Traces) │    │ (React)     │
                   └─────────┘    └─────────┘    └─────────────┘
```

All components are containerized and managed via Docker Compose for easy deployment.

## Quick Start Guide

### 1. Start the Metrics Stack

```bash
# Clone the repository
git clone https://github.com/your-org/llm-agent-metrics.git
cd llm-agent-metrics

# Start the metrics stack
docker-compose -f docker-compose-metrics.yml up -d
```

### 2. Add Metrics to Your Code

```python
# Import the metrics client
from metrics_client import configure_metrics, trace_context, span_context, add_metric

# Configure the client (do this once at startup)
configure_metrics(service_name="llm_agent_core")

# Trace a user request
with trace_context(user_prompt="Tell me a joke about AI"):
    # Span a specific operation
    with span_context("query_classification"):
        # Your code here
        category = classify_query(user_prompt)
        
        # Record a metric
        add_metric("classification.confidence", confidence_score)
```

### 3. Access the Dashboard

Open your browser and navigate to:
- http://localhost:3000

You should see metrics and traces from your instrumented code.

## Technology Choices

### Why InfluxDB?

We chose InfluxDB as our time-series database for several reasons:

1. **Optimized for Time-Series Data**: InfluxDB is purpose-built for time-series data, with efficient storage and retrieval
2. **Powerful Query Language**: Flux allows for complex time-based queries and aggregations
3. **Built-in Downsampling**: Automatic data reduction over time saves storage and improves query performance
4. **Tag-Based Indexing**: Fast filtering and grouping by metadata

**Alternatives Considered**:
- **Prometheus**: Great for metrics, but less suitable for high-cardinality data
- **TimescaleDB**: Excellent PostgreSQL extension, but required more infrastructure
- **DuckDB**: Considered for analysis, but lacks time-series optimization and remote storage

### Why MongoDB?

For trace storage, we chose MongoDB because:

1. **Flexible Schema**: Traces and spans can have varying structures and metadata
2. **Document Model**: Natural fit for hierarchical trace data
3. **Query Performance**: Fast lookups and filtering on arbitrary fields
4. **Scaling**: Easy to scale for high write throughput

**Alternatives Considered**:
- **Elasticsearch**: Great for log search, but higher resource requirements
- **PostgreSQL (JSONB)**: Would work, but less optimized for this workload
- **DuckDB**: Excellent for analytics but not designed for operational storage

### Why FastAPI?

For our collector and API services, we chose FastAPI for:

1. **Performance**: One of the fastest Python frameworks
2. **Type Safety**: Pydantic models provide validation and documentation
3. **Async Support**: Efficient handling of many concurrent connections
4. **OpenAPI Docs**: Automatic API documentation

## Core Concepts

### Traces and Spans

The foundation of our observability is distributed tracing:

- **Trace**: Represents a complete request/response cycle (e.g., processing a user query)
- **Span**: Represents a single operation within a trace (e.g., classifying a query)

Spans can have parent-child relationships, forming a tree that shows the execution path.

```
Trace: "Process user query"
├─ Span: "classify_query"
├─ Span: "generate_code"
│  ├─ Span: "prompt_engineering"
│  └─ Span: "code_extraction"
└─ Span: "execute_code"
   ├─ Span: "setup_environment"
   └─ Span: "run_docker"
```

### Metrics

Metrics are numeric values that change over time. Key types of metrics in our system:

- **Duration Metrics**: How long operations take
- **Count Metrics**: How often operations occur
- **Gauge Metrics**: Current value of something (e.g., memory usage)

Metrics can have tags for additional context, allowing filtering and grouping.

### Logs and Variables

Our system also captures:

- **Logs**: Traditional log messages with context
- **Variables**: State of important variables at specific points in execution

## Implementation Guide

### Basic Instrumentation

Here's a more detailed example of instrumenting your code:

```python
from metrics_client import (
    configure_metrics, trace_context, span_context, 
    log_info, log_error, capture_variable,
    add_metric, spanned, timed
)

# Configure client once at startup
configure_metrics(
    service_name="llm_agent_core",
    collector_url="http://localhost:5001",
    batch_size=20,
    flush_interval=5
)

# Process a user query
def process_user_query(query):
    # Create a trace for the entire operation
    with trace_context(query, tags={"type": "user_query"}):
        try:
            # Span a specific operation
            with span_context("classify_request"):
                log_info("Classifying user request")
                classification = classify_request(query)
                capture_variable("classification", classification)
                add_metric("classification.confidence", classification.confidence)
            
            # Using the decorator syntax for common operations
            response = generate_response(query, classification)
            
            log_info("Successfully processed query")
            return response
        except Exception as e:
            log_error(f"Error processing query: {str(e)}")
            raise

# Using decorators for common patterns
@spanned(tags={"module": "response_generation"})
def generate_response(query, classification):
    # Function implementation
    return {"response": "This is a response"}

@timed(metric_name="database.query_time")
def fetch_from_database(query):
    # Function implementation
    return {"data": "Some data"}
```

### Advanced Configuration

For production deployments, you'll want to configure:

```python
# In your application startup code
configure_metrics(
    service_name="llm_agent_core",
    collector_url="http://metrics-collector.production.svc:5001",
    batch_size=100,  # Larger batch for production
    flush_interval=2,  # More frequent flushing
    disable=os.getenv("DISABLE_METRICS", "false").lower() == "true"  # Environment toggle
)
```

### Using the API Directly

For custom use cases, you can use the MetricsClient directly:

```python
from metrics_client import MetricsClient, MetricsClientConfig

# Custom configuration
config = MetricsClientConfig(
    collector_url="http://localhost:5001",
    service_name="custom_service",
    batch_size=50,
    flush_interval=3,
    disable=False
)

# Get the client instance
client = MetricsClient.get_instance(config)

# Use the client directly
trace_id = client.start_trace("Custom trace")
span_id = client.start_span("Custom span")
client.add_metric("custom.metric", 42.0)
client.log("INFO", "Custom log message")
client.end_span(span_id)
client.end_trace(trace_id)
```

## UX Endpoints & Documentation

### Dashboard

The dashboard provides several views:

1. **Overview**: Summary of key metrics and recent traces
2. **Traces**: List and search all traces with filtering
3. **Spans**: Explore execution details within traces
4. **Metrics**: Interactive graphs of performance metrics
5. **Logs**: Search and filter logs from all services

### API Endpoints

The API Gateway provides these key endpoints:

#### Metrics Endpoints

- `GET /api/v1/metrics` - List available metrics
- `POST /api/v1/metrics/query` - Query time-series metrics

#### Trace Endpoints

- `GET /api/v1/traces` - List traces with filtering
- `GET /api/v1/traces/{trace_id}` - Get details of a specific trace
- `GET /api/v1/flame-graph/{trace_id}` - Get flame graph data for a trace

#### Span Endpoints

- `GET /api/v1/spans` - List spans with filtering
- `GET /api/v1/spans/{span_id}` - Get details of a specific span

#### Log Endpoints

- `GET /api/v1/logs` - List logs with filtering

#### Metadata Endpoints

- `GET /api/v1/services` - List all services
- `GET /api/v1/modules` - List all modules
- `GET /api/v1/functions` - List all functions

### API Documentation

Full OpenAPI documentation is available at:
- http://localhost:8000/docs

## FAQ

### General Questions

#### Q: How much overhead does instrumentation add?
**A:** The client library is designed to be lightweight with minimal impact on performance. Batching and background flushing ensure metrics collection doesn't block your main code path. In our testing, the overhead is typically less than 1% for most workloads.

#### Q: Can I use this in production?
**A:** Yes, the system is designed for production use. For high-scale deployments, you may want to scale the MongoDB and InfluxDB components horizontally.

#### Q: How much disk space do I need?
**A:** This depends on your volume of traces and metrics. InfluxDB has built-in downsampling which reduces storage needs over time. As a rule of thumb, start with 20GB of storage for each database and monitor usage.

### Technical Questions

#### Q: Why two databases instead of one?
**A:** InfluxDB is optimized for time-series data (metrics) while MongoDB is better suited for document storage (traces and spans). Using the right tool for each job provides better performance and functionality.

#### Q: Can I use DuckDB instead?
**A:** DuckDB is an excellent analytical database, but it's not designed for the concurrent write workloads that our metrics system requires. It could potentially be used for offline analysis of exported data, but not as a replacement for InfluxDB or MongoDB in the core system.

#### Q: How do I secure the metrics endpoints?
**A:** In production, you should:
1. Run the system behind a reverse proxy with authentication
2. Configure API keys for the collector service
3. Use network policies to control access between components

### Troubleshooting

#### Q: No metrics showing up in the dashboard?
**A:** Check:
1. Is the metrics collector running? (`docker-compose ps`)
2. Is your client configured with the correct collector URL?
3. Have you flushed metrics or waited for the automatic flush?
4. Check logs of the collector service for errors

#### Q: High latency in metrics collection?
**A:** Consider:
1. Increasing batch size to reduce HTTP requests
2. Checking network latency between your application and the collector
3. Scaling up the collector service if it's CPU-bound

## Research Topics

Interested in diving deeper? Here are some research topics related to our metrics system:

### 1. Distributed Tracing Systems

- **OpenTelemetry**: [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- **Jaeger**: [Jaeger Tracing](https://www.jaegertracing.io/)
- **Zipkin**: [Zipkin](https://zipkin.io/)

Our system is inspired by these industry-standard approaches but simplified for LLM Agent use cases.

### 2. Time-Series Databases

- **InfluxDB**: [InfluxDB Documentation](https://docs.influxdata.com/)
- **TimescaleDB**: [TimescaleDB](https://www.timescale.com/)
- **Prometheus**: [Prometheus](https://prometheus.io/)

Each has different strengths and trade-offs worth understanding.

### 3. Observability Concepts

- **Observability vs. Monitoring**: [Honeycomb's Explanation](https://www.honeycomb.io/blog/observability-101-terminology-and-concepts)
- **Cardinality and High-Dimensional Data**: [High Cardinality in Metrics](https://www.timescale.com/blog/what-is-high-cardinality-how-do-time-series-databases-influxdb-timescaledb-compare/)
- **RED Method**: Rate, Errors, Duration - a pattern for monitoring services

### 4. Visualization Techniques

- **Flame Graphs**: [Brendan Gregg's Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- **Heatmaps**: [Why Heatmaps for Time-Series Data](https://grafana.com/blog/2020/06/23/how-to-visualize-prometheus-histograms-in-grafana/)
- **Service Maps**: [Service Maps in Distributed Systems](https://www.datadoghq.com/blog/service-map/)

## Strategy Guide

### Onboarding to the Metrics Service

#### Step 1: Understanding the Basics (Week 1)

- Set up the metrics stack locally using Docker Compose
- Instrument a simple script with traces and spans
- Explore the dashboard to see your metrics
- Review the core concepts in this README

**Goal**: Get a basic understanding of how metrics flow through the system

#### Step 2: Integration with Your Code (Week 1-2)

- Identify key entry points in your application for tracing
- Start with coarse-grained instrumentation (main request flows)
- Add metrics for critical performance indicators
- Verify data is flowing correctly in the dashboard

**Goal**: Get basic monitoring of your application's core functionality

#### Step 3: Comprehensive Instrumentation (Week 2-3)

- Add finer-grained spans for important operations
- Capture contextual information with tags
- Record variable states at critical points
- Configure appropriate batch sizes and flush intervals

**Goal**: Get detailed visibility into your application's behavior

#### Step 4: Dashboard Customization (Week 3-4)

- Create custom dashboards for your specific needs
- Set up alerts for important metrics
- Build flame graph visualizations for common request types
- Document your instrumentation for the team

**Goal**: Make the metrics system work for your specific use cases

### Best Practices

1. **Start Broad, Then Refine**
   - Begin with high-level traces of major operations
   - Add more detailed spans as you identify bottlenecks
   - Don't try to instrument everything at once

2. **Use Meaningful Names**
   - Traces: Use the actual user query or request identifier
   - Spans: Name after the operation being performed
   - Metrics: Use a consistent naming scheme (e.g., `service.operation.measure`)

3. **Add Context with Tags**
   - Add tags for filtering and grouping (e.g., `environment`, `user_type`)
   - Keep high-cardinality data (like user IDs) out of tags
   - Use consistent tag names across services

4. **Be Privacy-Conscious**
   - Don't include PII in traces or metrics
   - Sanitize user queries before using as trace names
   - Use variable capture selectively for sensitive data

5. **Measure What Matters**
   - Focus on metrics that drive actions
   - Track both technical (e.g., latency) and business metrics (e.g., query success rate)
   - Set up baseline measurements before optimizing

### Getting Help

If you need help with the metrics system:

1. Check this README and the API documentation
2. Explore example code in the `examples/` directory
3. Reach out to the platform team on Slack (#metrics-help)
4. File issues on GitHub for bugs or feature requests

Remember, the goal of this system is to make your life easier by providing visibility into your LLM Agent's behavior. If it's not serving that purpose, we want to know!

---

Happy monitoring! 📊✨