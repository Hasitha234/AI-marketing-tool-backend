import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logging import get_logger

logger = get_logger("http")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Log request
        request_id = request.headers.get("X-Request-ID", "N/A")
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "N/A",
                "user_agent": request.headers.get("user-agent", "N/A")
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time * 1000, 2)
                }
            )
            
            return response
            
        except Exception as e:
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise 