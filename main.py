import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from crewai import Agent, Task, Crew, Process
# We will use crewai's new simplified LLM Integration
from crewai.llm import LLM

# -- Part 1: Configure the LLM ---
# This tells CrewAI how to connect to our local Ollama model
#The format 'ollama/<model_name>' is used to specify the model to be used. Replace <model_name> with the actual name of your Ollama model.
llm = LLM(
    model="ollama/llama3"
)

# Part 2: Define your agents (The team members) ---
# An agent is like an employeee with a specific job description

# Agent 1: The researcher

researcher = Agent(
    role="Senior Researcher",
    goal="Uncover ground breaking technologies and trens in {topic}",
    backstory=(
        "You distill the most relevant information from a large amount of data and provide concise summaries. "
        "You identify key trends and insights in the field of {topic} and provide actionable recommendations."
    ),
    verbose=True, # to see the agents thinking process
    allow_delegation=False, # this agent works alone on its tasks
    llm=llm #Asign the LLM to the agent so it can use it to perform its tasks
)

# Agent 2: The writer

writer = Agent(
    role="Technical Writer",
    goal="Create a compelling and insightful blog on {topic}",
    backstory=(
        "You have a knack for turning complex technical concepts into easy-to-understand content. "
        "You collaborate with researchers and engineers to ensure accuracy and clarity in all documentation."
    ),
    verbose=True, # to see the agents thinking process
    allow_delegation=False, # this agent works alone on its tasks
    llm=llm #Asign the LLM to the agent so it can use it to perform its tasks
)

# task 1: Research the topic
research_task = Task(
    description="Conduct in-depth research on {topic} and provide a comprehensive summary of findings.",
    expected_output="A bulleted list summary of the top 3-5 trends in the specified topic {topic}.",
    agent=researcher # asign this task to the researcher agent
)

# task 2: Write the blog
write_task = Task(
    description=("Write a blog post based on the research findings provided by the researcher. "
                 "Include: a catchy title, an engaging introduction, a detailed body with insights and trends, and a conclusion with actionable takeaways."),
    expected_output="A well-structured blog post that effectively communicates the key trends and insights in the specified topic {topic}.",
    agent=writer, # asign this task to the writer agent
    context=[research_task]  # This is the MAGIC task depends on the output of the research task
)   

#form the crew with the agents and tasks
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True# see the crew's thinking process
)

if __name__ == "__main__":
    # Run the crew with a specific topic
    topic = "Artificial Intelligence in Healthcare"

    #start the crew's work on the specified topic
    results = crew.kickoff(inputs={"topic": topic})
    ## or crew.run(topic=topic)
    # print the final results of the crew's work
    print("\nFinal Results:")
    print(results)