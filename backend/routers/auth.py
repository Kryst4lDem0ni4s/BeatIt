import logging
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends
import firebase_admin
from firebase_admin import credentials, auth, db
from fastapi import Depends, FastAPI, HTTPException, status, APIRouter
from fastapi import Request
from models.model_types import LoginRequest

app = FastAPI()
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register")
async def register(request: Request):
    """
    Register a new user with Firebase Authentication.
    """
    logger.info("Register request received.")
    try:
        data = await request.json()
        logger.info(f"Register request received: {data}")

        # Extract verification data
        verification_data = data.get("signUpRequest", {})
        if not verification_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="verification_request is required"
            )
        
        email = verification_data.get("email")
        password = verification_data.get("password")
        firstname = verification_data.get("firstname")
        lastname = verification_data.get("lastname")
        username = verification_data.get("username")
        phone_number = verification_data.get("phone_number")

        logger.info("Registering user: %s", email)

        # Create user in Firebase Authentication
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=f"{firstname} {lastname}",
            phone_number=phone_number if phone_number else None
        )

        # Store additional user data in Realtime Database
        user_data = {
            "email": email,
            "firstname": firstname,
            "lastname": lastname,
            "username": username,
            "phone_number": phone_number,
            "created_at": {".sv": "timestamp"}
        }

        # Use the reference method from db to set data
        db.reference(f"users/{user_record.uid}").set(user_data)
        logger.info("User registered successfully: %s, for uid: %s", user_data, user_record.uid)
        
        # Generate a custom token for the user using the UID
        try:
            custom_token = auth.create_custom_token(user_record.uid)
        except Exception as e:
            logger.error("Error generating custom token: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate authentication token.") from e
        
        token_string = custom_token
    
        logger.info("User custom token registered successfully: %s", token_string)

        return {
            "status": "success",
            "message": "User registered successfully",
            "email": email,
            "token": token_string
        }

    except auth.EmailAlreadyExistsError as exc:
        logger.error("Email already in use")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already in use"
        ) from exc

    except auth.PhoneNumberAlreadyExistsError as exc:
        logger.error("Phone number already in use")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone number already in use"
        ) from exc

    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        ) from e

@router.post("/login")
async def login(login_request: LoginRequest):
    try:
        email = login_request.email
        password = login_request.password

        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password are required"
            )

        logger.info(f"Login attempt for email: {email}")
        
        try:
            user = auth.get_user_by_email(email)
        except auth.UserNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            ) from exc

        
        # Create a custom token for the user
        try:
            custom_token = auth.create_custom_token(user.uid)
        except Exception as e:
            logger.error(f"Error generating custom token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate authentication token"
            ) from e

        # Fetch user details from Realtime Database
        user_data_ref = db.reference(f"users/{user.uid}")
        user_data = user_data_ref.get()

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User data not found in database"
            )

        logger.info("User logged in successfully: %s", user.uid)

        return {
            "status": "success",
            "message": "Login successful",
            "user": user_data,
            "token": custom_token
        }

    except HTTPException as exc:
        logger.error("Login failed: %s", exc.detail)
        raise exc

    except Exception as e:
        logger.error("Unexpected error during login: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        ) from e

@router.post("/forgot-password")
async def forgot_password(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )

        # Generate password reset link
        reset_link = auth.generate_password_reset_link(email)
        
        # TODO: Implement email sending logic here
        logger.info(f"Password reset link generated: {reset_link}")
        
        # Always return success to prevent email enumeration
        return {
            "status": "success",
            "message": "If an account exists, a password reset email has been sent"
        }

    except auth.UserNotFoundError:
        # Log but still return success message
        logger.info(f"Password reset requested for non-existent email: {email}")
        return {
            "status": "success",
            "message": "If an account exists, a password reset email has been sent"
        }
        
    except Exception as e:
        logger.error(f"Password reset failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        ) from e

@router.post("/logout")
async def logout(request: Request):
    try:
        data = await request.json()
        id_token = data.get("id_token")
        
        if not id_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ID token is required"
            )

        # Verify ID token and get UID
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        
        # Revoke all refresh tokens
        auth.revoke_refresh_tokens(uid)
        
        logger.info(f"User logged out: {uid}")
        return {
            "status": "success",
            "message": "Successfully logged out"
        }

    except auth.ExpiredIdTokenError as e:
        logger.warning(f"Expired ID token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        ) from e
        
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid ID token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        ) from e
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        ) from e

@router.post("/user")
async def user_specific_endpoint(request: Request):
    try:
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        id_token = auth_header.split("Bearer ")[1]
        
        # Verify the token
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            
            # Now you can use uid to:
            # 1. Fetch user-specific data
            user_data = db.reference(f"users/{uid}").get()
            
            # 2. Assign items to this specific user
            # 3. Return customized data based on user preferences
            
            return {
                "status": "success",
                "user_data": user_data,
                # "customized_content": get_customized_content(uid)
            }
            
        except auth.ExpiredIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except auth.InvalidIdTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
    except Exception as e:
        logger.error(f"Error in user-specific endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process request"
        )

bearer_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        token = credentials.credentials
        # Verify the Firebase token
        decoded_token = auth.verify_id_token(token)
        # Return user data
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name")
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
