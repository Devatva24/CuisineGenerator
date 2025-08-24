from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from secret_key import groq_api_key
import os

# Set API key
os.environ["GROQ_API_KEY"] = groq_api_key

# Create the LLM (Groq uses chat models, but same interface)
llm = ChatGroq(
    model="llama3-8b-8192",  # you can also try "llama3-70b-8192"
    temperature=0.7
)

def generate_restaurant_name_and_items(cuisine):
    # Prompt 1: Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    # Prompt 2: Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string"
    )

    # Step 1: cuisine → restaurant_name
    name_chain = prompt_template_name | llm
    restaurant_name = name_chain.invoke({"cuisine": cuisine}).content.strip()

    # Step 2: restaurant_name → menu_items
    menu_chain = prompt_template_items | llm
    menu_items = menu_chain.invoke({"restaurant_name": restaurant_name}).content.strip()

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Italian"))
