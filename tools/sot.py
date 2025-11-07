def get_skeleton_prompt(question:str):
    # Prepare the prompt for generating skeleton.
    prompt =  "You're an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, each skeleton point should be very short with only 3~5 words. Generally, the skeleton should have 3~10 points.\n\nQuestion:\nWhat are the typical types of Chinese dishes?\nSkeleton:\n1. Dumplings. \n2. Noodles. \n3. Dim Sum. \n4. Hot Pot. \n5. Wonton. \n6. Ma Po Tofu. \n7. Char Siu. \n8. Fried Rice. \n\nQuestion:\nWhat are some practical tips for individuals to reduce their carbon emissions?\nSkeleton:\n1. Energy conservation. \n2. Efficient transportation. \n3. Home energy efficiency. \n4. Reduce water consumption. \n5. Sustainable diet. \n6. Sustainable travel. \n\nNow, please provide the skeleton for the following question.\n{question}\nSkeleton:\n ".format(question=question)
    return prompt
def get_point_expanding_prompt(skeleton:str, question:str):
    # Prepare the prompt for expanding points.
    points = skeleton.split("\n")
    prompts_for_points = []
    shared_perfix = """[INST] You're responsible for continuing the writing of one and only one point in the overall answer to the following question.\n\n{question}\n\nThe skeleton of the answer is\n\n{skeleton}\n\n Write it **very shortly** in 1~2 sentence and do not continue with other points! Continue and only continue the writing of point """.format(question=question,skeleton=skeleton)
    # shared_perfix = """[INST] You're responsible for expanding only the specific point mentioned in the skeleton of the following question.\n\n{question}\n\nThe skeleton of the answer is\n\n{skeleton}\n\n Focus solely on the specified point and **write briefly** in 1~2 sentences without moving to any other points. Continue and only expand the writing of point  """.format(question=question,skeleton=skeleton)

    # Get points and prompts for each point.
    for idx, i in enumerate(points):
        prompt = "{point}. [/INST]".format(point=idx+1)
        prompts_for_points.append(prompt + i)
    return points,shared_perfix,prompts_for_points

