from retrieval import chain
while {True}:
  str = input("Enter your Question : ")
  if str.lower() == "/quit" or str.lower() == "/q":
    break
  print(chain.invoke(str))

# vector_db.delete_collection()