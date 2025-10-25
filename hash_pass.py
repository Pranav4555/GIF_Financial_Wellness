import streamlit_authenticator as stauth

# Hash the password "pranav4554"
passwords = ["pranav4554"]
hashed_passwords = stauth.Hasher(passwords).generate()

print("Your hashed password:")
print(hashed_passwords[0])
