from agents import Agent,Runner,AsyncOpenAI , OpenAIChatCompletionsModel , function_tool , handoffs
from agents.run import RunConfig 
import os
from dotenv import load_dotenv , find_dotenv



@function_tool
def get_flights(destination: str) -> list:
    sample_flights = {
        "Karachi": [
            "PIA PK301 - ISB to KHI - 2h - Rs. 15,000",
            "AirSial SAI401 - LHE to KHI - 2h - Rs. 13,500",
            "SereneAir ER504 - PEW to KHI - 2h 10m - Rs. 14,200"
        ],
        "Lahore": [
            "PIA PK302 - KHI to LHE - 1h 45m - Rs. 16,000",
            "Airblue PA407 - ISB to LHE - 1h - Rs. 12,800",
            "Fly Jinnah 9P711 - KHI to LHE - 2h - Rs. 13,000"
        ],
        "Islamabad": [
            "PIA PK351 - KHI to ISB - 2h - Rs. 14,500",
            "SereneAir ER506 - LHE to ISB - 1h - Rs. 12,500",
            "Airblue PA200 - PEW to ISB - 50m - Rs. 10,000"
        ],
        "Paris": [
            "Air France AF007 - JFK to CDG - 9h - $620",
            "Delta DL121 - LAX to CDG - 11h - $580",
            "Emirates EK73 - DXB to CDG - 7h - $700"
        ],
        "Tokyo": [
            "ANA NH105 - LAX to HND - 11h - $750",
            "Japan Airlines JL43 - SFO to NRT - 10.5h - $720",
            "Qatar Airways QR806 - DOH to NRT - 10h - $690"
        ],
        "Dubai": [
            "Emirates EK601 - KHI to DXB - 2h 15m - Rs. 85,000",
            "Flydubai FZ336 - LHE to DXB - 3h - Rs. 80,000",
            "PIA PK233 - ISB to DXB - 3h 10m - Rs. 78,000"
        ],
        "Bali": [
            "Singapore Airlines SQ948 - SIN to DPS - 2.5h - $320",
            "Garuda GA715 - SYD to DPS - 6h - $540",
            "Emirates EK398 - DXB to DPS - 9h - $810"
        ]
    }
    return sample_flights.get(destination, [f"No flights found for {destination}"])


@function_tool
def suggest_hotels(destination: str) -> list:
    sample_hotels = {
        "Karachi": [
            "Mövenpick Hotel – 5⭐ – Rs. 27,000/night – Central Clifton area",
            "Avari Towers – 4⭐ – Rs. 19,000/night – Shahrah-e-Faisal",
            "Regent Plaza – 3⭐ – Rs. 9,000/night – Good for business stays"
        ],
        "Lahore": [
            "Pearl Continental – 5⭐ – Rs. 22,000/night – Near Mall Road",
            "The Nishat Hotel – 4⭐ – Rs. 18,000/night – Gulberg area",
            "Faletti's Hotel – 3⭐ – Rs. 10,000/night – Historic and elegant"
        ],
        "Islamabad": [
            "Serena Hotel – 5⭐ – Rs. 25,000/night – Top luxury stay",
            "Hotel Margala – 4⭐ – Rs. 14,000/night – Near Convention Center",
            "Envoy Continental – 3⭐ – Rs. 8,500/night – Blue Area"
        ],
        "Paris": [
            "Hotel Le Meurice – 5⭐ – $950/night – Luxury near Louvre",
            "Hôtel Fabric – 4⭐ – $220/night – Trendy boutique hotel",
            "CitizenM Paris – 3⭐ – $140/night – Budget and modern"
        ],
        "Tokyo": [
            "Park Hyatt Tokyo – 5⭐ – $820/night – Famous from Lost in Translation",
            "Hotel Gracery Shinjuku – 4⭐ – $200/night – Godzilla-themed",
            "Sakura Hotel Jimbocho – 3⭐ – $80/night – Affordable and central"
        ],
        "Dubai": [
            "Burj Al Arab – 7⭐ – $2000/night – Ultimate luxury",
            "Atlantis The Palm – 5⭐ – $850/night – Beach + Waterpark",
            "Rove Downtown – 3⭐ – $120/night – Best budget hotel"
        ],
        "Bali": [
            "Four Seasons Resort – 5⭐ – $880/night – Ocean-view villas",
            "Alaya Resort Ubud – 4⭐ – $210/night – Near rice terraces",
            "Maha House – 3⭐ – $60/night – Cozy traditional stay"
        ]
    }
    return sample_hotels.get(destination, [f"No hotels found for {destination}"])
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")





provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,

)

run_config = RunConfig(
    model = model,
    model_provider = provider,
    tracing_disabled = True,
)



print("#################### Welcome to AI driven Travel Agent! ###########################")
print('''What would you like to do today? 
      1. You can ask for hotel suggestions or flight information for a specific destination.
      2. You can ask for foods or activities in a specific destination.
      3. You can ask for a book complete travel plan for a specific destination.''')
user_input = input("Please enter your query: ").strip().lower()

if user_input:
    
        dest_input = input("Please enter your desired destination (e.g., Karachi, Lahore, Paris): ").strip().capitalize()
        

    
BookingAgent = Agent(
        name="Booking Agent",
        instructions='''You are a helpful agent that can book hotels and flights for the user based on their destination input.
        You will use the tools provided to suggest hotels and flights.''',
        tools=[suggest_hotels, get_flights],
)

DestinationAgent = Agent(
    name="Destination Agent",
    instructions='''You are a helpful agent that suggest places for the user to explore and plan their trip
    based on the user's destination input.''',
    tools=[suggest_hotels, get_flights]
     
)


ExploreAgents = Agent(
        name="Explore Agent",
        instructions='''You are a helpful agent that can suggest foods and activities for the user based on their destination input.''',
        tools=[suggest_hotels, get_flights],
)



if user_input == "1":
        run = Runner.run_sync(
                starting_agent=BookingAgent,
                run_config=run_config,
                input=dest_input,
)
elif user_input == "2":
        run = Runner.run_sync(
                starting_agent=ExploreAgents,
                run_config=run_config,
                input=dest_input,
)
elif user_input == "3":
        run = Runner.run_sync(
                starting_agent=DestinationAgent,
                run_config=run_config,
                input=dest_input,
)

print(run.final_output)