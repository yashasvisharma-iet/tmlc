from crewai import Agent, Task, Crew
import ollama

class EmailDrafter:
    def __init__(self):
        self.drafter = Agent(
            role="Email Drafter",
            goal="Compose clear, concise, and effective response emails for any purpose.",
            backstory="You're a highly skilled assistant specializing in drafting professional response emails."
        )

    def run(self, input_email):
        description = f"""
        Draft a professional response email based on the provided input.

        Input Email:
        {input_email}
        """

        task = Task(
            description=description,
            agent=self.drafter,
            expected_output="A well-structured response email."
        )

        crew = Crew(agents=[self.drafter], tasks=[task])

        # Generate response using Ollama
        response = ollama.chat(model="phi3", messages=[
            {"role": "system", "content": "You are an AI email assistant that drafts professional emails."},
            {"role": "user", "content": description}
        ])

        return response["message"]["content"]

# Sample email input
sample_email = """
Dear A,

I hope this message finds you well. I wanted to take a moment to express my heartfelt gratitude for your incredible support during our recent project at XYZ Solutions Inc.

Your dedication, expertise, and proactive approach made a significant difference in achieving our goals. Your contributions ensured the project was not only completed on time but also exceeded expectations. I truly appreciate your attention to detail and your collaborative spirit.

Working with you on this project has been an absolute pleasure. If there’s ever an opportunity for me to return the favor or support you in any way, please don’t hesitate to let me know.

Thank you once again for your invaluable contributions!

Warm regards,  
B  
Senior Project Manager  
XYZ Solutions Inc.  
b@XYZ.com  
"""

# Run the email drafter
crew = EmailDrafter()
result = crew.run(sample_email)

# Print the generated email response
print("\nGenerated Email:\n", result)
