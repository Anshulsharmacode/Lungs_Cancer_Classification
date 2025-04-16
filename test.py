class A:
    token:str
    def __init__(self,token ):
        self.__token = token 
        
    def get_token (self):
        return self.__token 

    def set_token(self,en):
        self.__token = en
        return self.__token


ob = A("anshul")

print(ob.token)

# fu = ob.get_token()
# print(fu)

# obb = A("kannu")

# updat = obb.set_token("kannu")
# print(updat)


    # def set_token()

# 1. Create a clas with a variable "token"
# 2. Create GetSet functions
# 3. Intantiate that class and initialize with "anshul"
# 4. print the token
# 5. Set the token to "kanu"
# 6. print token

# condition:
# can't directly access token.add(element)


