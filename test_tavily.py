# VeriRAG Tavily Search Test
#
# This test verifies that:
# 1. Tavily is configured correctly.
# 2. The web_search() function returns a list.
# 3. Search results contain the expected structure.


from app.tools import web_search


# Run a test web search.

query = "Mahatma Gandhi birth date"

results = web_search(query)


print("QUERY:")
print(query)


print("\nTYPE:")
print(type(results))


print("\nNUMBER OF RESULTS:")
print(len(results))


# Safely inspect the first result.

if results:

    print("\nFIRST RESULT TYPE:")
    print(type(results[0]))

    print("\nFIRST RESULT:")
    print(results[0])

else:

    print("\nNO RESULTS FOUND.")