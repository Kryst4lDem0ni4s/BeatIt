import logging
import firebase_admin
from firebase_admin import credentials, auth, db
from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi import Request
from backend.models.model_types import LoginRequest


# Initialize Firebase Admin SDK
# cred = credentials.Certificate("path/to/your/firebase/credentials.json")
# firebase_admin.initialize_app(cred)

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

