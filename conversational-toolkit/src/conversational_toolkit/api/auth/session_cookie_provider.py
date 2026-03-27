from abc import ABC

from fastapi import APIRouter, HTTPException, Request, Response, status
from jose import JWTError, jwt  # type: ignore[import-untyped]

from conversational_toolkit.api.auth.base import AuthProvider
from conversational_toolkit.conversation_database.controller import ConversationalToolkitController
from conversational_toolkit.utils.database import generate_uid


class SessionCookieProvider(AuthProvider, ABC):
    def __init__(
        self,
        controller: ConversationalToolkitController,
        secret_key: str = "1234567890",
        algorithm: str = "HS256",
        env: str = "local",
    ):
        self.controller = controller
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.env = env
        self.cookie_name = "access_token"

    def bind_to_app(self, app):
        router = APIRouter(prefix="/auth")

        @router.post("/refresh")
        async def refresh(
            request: Request,
            response: Response,
        ) -> None:
            access_token = request.cookies.get(self.cookie_name)

            if not access_token:
                user = await self.controller.register_user(user_id=generate_uid())
                response.set_cookie(
                    key=self.cookie_name,
                    value=jwt.encode({"sub": user.id}, self.secret_key, algorithm=self.algorithm),
                    expires=365 * 24 * 60 * 60,
                    httponly=True,
                    secure=False if self.env == "local" else True,
                )

        app.include_router(router)

    def get_current_user_id(self, request: Request) -> str:
        token = request.cookies.get(self.cookie_name)
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

        try:
            claims = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            sub = claims.get("sub")
            if sub is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject claim")
            return str(sub)
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
