import pandas as pd
import openai


def query_interview_basic_iters(filename, query, api_key): 

    df = pd.read_csv(filename)

    answers = []

# process df in chunks
    chunk_size = 100
    chunk_up = 95 
    for chunk_start in range(0, len(df), chunk_up):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]

        answer = query_dataframe(chunk, query, api_key)

        if 'NNN///AAA' not in answer and "'NNN///AAA'" not in answer: 
            answers.append(answer)
            print(f"Chunk:[{chunk_start}, {chunk_end}], Answer: {answer}\n")
        else: 
            print(f"SKIPPED Chunk:[{chunk_start}, {chunk_end}], Answer: {answer}\n")

    
    return answers 

def query_interview_basic(filename, query, api_key, summary_query=None): 

    q = query + """Give the exact original row number and the context in which this is discussed, and print out the exact line. If the topic in question 
        is not discussed, then return only the string 'NNN///AAA' and nothing else; only do this if it is certain the topic in question was not discussed."""

    answers = query_interview_basic_iters(filename, q, api_key)
    
    answer_joined = '\n\n'.join(answers)

    if not summary_query: 
        summary_query = f'''Each of these statements describes a subset chunk of an interview. Combine all of this information into one summary answer of 
            the question "{query}" for the entire interview. Give the exact original row number, the context of discussion, and the exact line, for every single 
            original row number mentioned as having relevant information. Omit as little information as possible. '''


    response = openai.chat.completions.create(
        model="gpt-4-turbo", 
        messages=[
            {"role": "system", "content": f"Statements:{answer_joined}\nQuestion: {summary_query}\n"},
            {"role": "user", "content": ""}
        ]
    )
    final_answer = response.choices[0].message.content

    return final_answer 




def query_dataframe(df, query, api_key):
    openai.api_key = api_key

    json_data = df.to_json(orient='split')

    response = openai.chat.completions.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given the following data: {json_data}"},
            {"role": "user", "content": f"Answer the query: {query}"}
        ]
    )
    answer = response.choices[0].message.content 
    return answer

def query_interview_multiple(filenames, query): 
    answerdict = {}

    for filename in filenames: 
        answerdict[filename] = query_interview_basic(filename, query)

    return answerdict 


def summarize_answers(answerdict, query): 
    
    dict_str = ''
    for key,val in answerdict.items(): 
        dict_str += f"\n{key}: {val}"

    question = f'''Each of these statements describes one interview. Based on these statements, what is the answer to the question
         "{query}" for all of the interviews?'''

    
    response = openai.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": f"Statements:{dict_str}\nQuestion: {question}\n"},
            {"role": "user", "content": ""}
        ]
    )

    final_answer = response.choices[0].message.content

    return final_answer 


def verify_answer(filename, verification_query, api_key, summary_query): 
    verification_response = query_interview_basic(filename, verification_query, api_key, summary_query=summary_query)
    return verification_response


def get_row_nums(response, api_key): 
    query = f"Extract all the row numbers mentioned as discussing friendship in this TEXT, and print them in an array. \nTEXT: {response}"
    openai.api_key = api_key

    response = openai.chat.completions.create(
    model="gpt-4-turbo", 
    messages=[
        {"role": "system", "content": query},
        {"role": "user", "content": ""}
        ]
    )
    final_answer = response.choices[0].message.content

    return final_answer


def add_boolean_col(filename, download_path, new_col_name, response, api_key, new_name=None): 
    row_num_str = get_row_nums(response, api_key)
    print(row_num_str)

    row_nums = [int(x) for x in row_num_str[1:-1].split(', ')]

    df = pd.read_csv(filename)
    
    df[new_col_name] = 0 
    df.loc[row_nums, new_col_name] = 1 
    
    if not new_name: 
        new_name = filename.split('/')[-1][:-4] + '_NEW'

    df.to_csv(download_path + new_name + '.csv')


def query_and_add_bool_col(filename, query, api_key, download_path, new_col_name): 

    response = query_interview_basic(filename, query, api_key)
    print(response)

    add_boolean_col(filename, download_path, new_col_name, response, api_key)


#filename = 'interview_csvs/ZMU_New_NY_4November2022.csv'
#f = 'BTK_Bay_SoCal_25August2022'
#f = 'BZR_New_SoCal_5August2022'
# f = 'JHN_Los_SoCal_10August2022'

filename = 'interview_csvs/' + f + '.csv'

api_key = #api key here

#quality time 
q_query = """I am interested in finding instances from the interview transcript where the participant talked about friendship as spending quality time with 
    others, meaning that they chose to spend time together. Find all instances in the attached transcript in which the participant talked about developing 
    friendships with others by spending quality time together."""


i_query = """"I am interested in finding instances from the interview transcript where the participant talked about friendship as an interest-driven practice, 
    meaning that friendship develops when people share similar interests such as in making music or painting. Find all instances in the attached transcript 
    in which the participant talked about developing friendships with others around shared interests."""

response = query_interview_basic(filename, i_query, api_key)
print('FINAL ANSWER\n', response)

# verification_query = f"""You will be given a range of data (indicated by the original row numbers) and a set of statements corresponding to specific row numbers. 
#     From the set of statements, select only the statements that have corresponding row numbers within the range of the given data, and verify the correctness of 
#     only those in-range statements. Do not comment on any out-of-range-statements at all. \nSet of statements to be verified: {response}"""

# summary_query =  f"""Based on the given statements, is the following INFORMATION correct? \nINFORMATION= {response}"""

# verification_response = verify_answer(filename, verification_query, api_key, summary_query=summary_query)

# print('VERIFICATION\n', verification_response)

# row_nums = get_row_nums(response, api_key)
# print(row_nums)

download_path = #download destination here
new_col_name = 'Friendship Discussed'
#newname = f + '_quality_time_only'
newname = f + '_interest_driven'

add_boolean_col(filename, download_path, new_col_name, response, api_key, new_name=newname)
print('DONE')