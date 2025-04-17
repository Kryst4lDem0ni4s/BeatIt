from pydantic import BaseModel, Field

# class SignUpRequest(BaseModel):
#     email: str
#     username: str
#     password: str
#     phonenumber: str = Field(..., pattern=r'^\+91\d{10}$')

class LoginRequest(BaseModel):
    email: str
    password: str
    
# class EmailRequest(BaseModel):
#     email: str
    
# class UpdatePasswordRequest(BaseModel):
#     uid: str
#     new_password: str

# class Profile(BaseModel):
#     fullname: str
#     password: constr(min_length=8)
#     phonenumber: str = Field(..., pattern=r'^\+91\d{10}$')
#     address: str
#     email: str
