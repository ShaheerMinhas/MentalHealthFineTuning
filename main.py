from gradientai import Gradient
import os
def main():
    os.environ['GRADIENT_ACCESS_TOKEN'] = "5nC8r5yEZ5ttVlWGihEvivPopkHCcL6J"
    os.environ['GRADIENT_WORKSPACE_ID'] = "d45e2602-2f9f-4d21-a21f-a4fdfc55d68e_workspace"
    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

        new_model_adapter = base_model.create_model_adapter(
            name="test model 3"
        )
        print(f"Created model adapter with id {new_model_adapter.id}")
        sample_query = "### Instruction: Suggest a therapist in Lahore ? \n\n### Response:"
        print(f"Asking: {sample_query}")    
        # before fine-tuning
        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated (before fine-tune): {completion}")    
        samples = [
    { "inputs": "### Instruction: What mental health resources are available in Lahore? \n\n### Response: Lahore offers various mental health clinics and support groups for individuals seeking assistance with their mental well-being." },
    { "inputs": "### Instruction: Suggest a therapist in Lahore ? \n\n### Response: To find a therapist in Lahore, you can search online directories .Some of centers for Mental Health are :   1.Holistic Cure Clinic   2.Jinnah Hospital    3.The Parklane Clinic" },
    { "inputs": "### Instruction: What is the cost of getting Mental Health Support  \n\n### Response: It can cost around 3 to 5k on average . You can also go for online sessions which can cost around 1 to 2 k per session on average" },
   

]






      # this is where fine-tuning happens
      # num_epochs is the number of times you fine-tune the model
      # more epochs tends to get better results, but you also run the risk of "overfitting"
      # play around with this number to find what works best for you
        num_epochs = 3
        count = 0
        while count < num_epochs:
            print(f"Fine-tuning the model, iteration {count + 1}")
            new_model_adapter.fine_tune(samples=samples)
            count = count + 1
        # after fine-tuning
        completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
        print(f"Generated (after fine-tune): {completion}")
        new_model_adapter.delete()

if __name__ == "__main__":
    main()