from django.http import HttpResponse


ALLOWED_CORS_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}


class DevCorsMiddleware:
    """
    Minimal development CORS support for the Vite frontend.

    This intentionally allows only the local frontend origins
    needed by the current project instead of opening all origins.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.headers.get("Origin")

        if request.method == "OPTIONS" and origin in ALLOWED_CORS_ORIGINS:
            response = HttpResponse(status=200)
        else:
            response = self.get_response(request)

        if origin in ALLOWED_CORS_ORIGINS:
            response["Access-Control-Allow-Origin"] = origin
            response["Vary"] = "Origin"
            response["Access-Control-Allow-Credentials"] = "true"
            response["Access-Control-Allow-Headers"] = (
                "Accept, Authorization, Content-Type, Origin, X-Requested-With"
            )
            response["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            )

        return response
