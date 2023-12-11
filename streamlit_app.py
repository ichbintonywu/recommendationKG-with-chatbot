import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from streamlit_option_menu import option_menu
from neo4j_handler import exec_cypher_query,exec_simple_cypher_write,exec_cypher_write,gdsrun
from itertools import chain
from streamlit_agraph import agraph, TripleStore, Node, Edge, Config
import numpy as np
import graphviz
import time
import openai
from datetime import datetime
import calendar
import pandas as pd
import altair as alt
from tabulate import tabulate
from recommengine import recommdation_engine,prediction_engine

API_KEY = "sk-**********************"
LAN_MODEL = "text-davinci-003"

st.set_page_config(layout="wide", page_title="main_page")
st.markdown("<h1 style='text-align: center; color: MediumVioletRed;'>Movie Knowledge Graph</h1>", unsafe_allow_html=True)
# --- NAVAGATION MENU ---
selected = option_menu(
    menu_title=None,
    options=["Dashboard","Input And Train to Recommend","Movie ChatBot"],
    icons=["flower2","building-fill-up","chat-heart"],
    orientation="horizontal"
)

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
            """
st.markdown(hide_st_style,unsafe_allow_html=True)

def get_neo4j_feedllm_results(search):
    training = """
        Follow the description to generate a query statement shown below.
        #What is the recommended movie for Keanu Reeves?
        MATCH (n:Person)-[:HAS_RECOMMENDATION]-(m:Recommendation) WHERE n.name =~ '(?i)Keanu Reeves' RETURN m.movies AS movie;

        #Do you have movie recommendation for Keanu Reeves?
        MATCH (n:Person)-[:HAS_RECOMMENDATION]-(m:Recommendation) WHERE n.name =~ '(?i)Keanu Reeves' RETURN m.movies AS movie;

        #Can you recommend a movie for Keanu Reeves?
        MATCH (n:Person)-[:HAS_RECOMMENDATION]-(m:Recommendation) WHERE n.name =~ '(?i)Keanu Reeves' RETURN m.movies AS movie;

        #Any prediction for Keanu Revees?
        MATCH (n:Person)-[:PREDICATED_RECOMMEND]-(m:Recommendation) WHERE n.name =~ '(?i)Keanu Reeves' RETURN m.movies AS movie;

        #Can you predict a movie for John Goodman?
        MATCH (n:Person)-[:PREDICATED_RECOMMEND]-(m:Recommendation) WHERE n.name =~ '(?i)John Goodman' RETURN m.movies AS movie;
        #;
"""
    query = training + search + "\n"

    start = time.time()
    openai.api_key = API_KEY
    model = LAN_MODEL
    try:
          response = openai.Completion.create(engine=model, prompt = query, max_tokens= 50,stop=["#", ";"],)
          generated_text = response.choices[0].text
          prompt_result = generated_text
        #   return prompt_result
          query_to_prompt =  exec_cypher_query(prompt_result)
        #   print (prompt_result)
          print(query_to_prompt)
          return query_to_prompt
    finally:
        print('Neo4j query generation time:', time.time()-start)

def get_results(messages):
    start = time.time()
    openai.api_key = API_KEY
    model = LAN_MODEL
    try:
          response = openai.Completion.create(engine=model, prompt=messages, max_tokens=150)
          generated_text = response.choices[0].text
          prompt_result = generated_text
          return prompt_result
    finally:
        print('LLM generation time:', time.time()-start)


with st.sidebar:
    st.title('Movie KG App')
    st.markdown('''
    ### About
    This app is an LLM-powered chatbot built using OpenAI and Neo4j
    ''')
    add_vertical_space(5)
    st.info('Made with ❤️')

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = hugchat.ChatBot()
    response = chatbot.chat(prompt)
    return response

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text



if selected == "Dashboard":
    print("Welcome to the Dashboard page")
    config = Config(height=600,
		            width=1200,
                    nodeHighlightBehavior=True,
                    highlightColor="#F7A7A6",
                    directed=True,
                    collapsible=True)
    nodes = []
    edges = []
    # # nodes.append(Node(id="Spiderman", size=15, label="Peter Parker", image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png") )
    # nodes.append( Node(id="Captain_Marvel", size=15,label="Person", image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png") )
    # edges.append( Edge(source="Captain_Marvel", target="Spiderman",label="LINK_TO", size=5,type="CURVE_SMOOTH"))
    # agraph(nodes, edges, config)

    moive_graph_query_str="""
    match (n:Recommendation)<-[r:HAS_RECOMMENDATION]-(p:Person)
    with '\n' AS separator,p.name as person_name,type(r) as rel_name, n.movies as movie_list limit 20
    with person_name,rel_name,REDUCE(mergedString = "",item IN movie_list |
              mergedString
              + CASE WHEN mergedString='' THEN '' ELSE separator END
              + item) AS movie_title
return person_name,rel_name,movie_title
    """
    movie_graph_pair =  exec_cypher_query(moive_graph_query_str)
    # print(movie_graph_pair)

    released_graph_query_str="""
    MATCH (m:Movie) with m.title as title,m.tagline as tabline,m.released as released return toInteger(released) as released, count(title) as count order by released desc
    """
    released_statistics =  exec_cypher_query(released_graph_query_str)

    highest_score_movie_query ="""
    match (n:Person)-[r:RATED]-(m:Movie)
    with m.title as title, avg(toInteger(r.score)) as average_rate
    return title as Movie,average_rate as Avg_Rate order by Avg_Rate desc limit 10
    """
    get_highest_score_movie= exec_cypher_query(highest_score_movie_query)

    most_watched_movie_query ="""
    MATCH p=(n:Person)-[r:WATCHED]->(m:Movie)
    with m.title as title, size(collect(r)) as watched
    return title,watched order by watched desc limit 10
    """
    get_most_watched_movie=exec_cypher_query(most_watched_movie_query)

    person_graph_query_str="""
    MATCH (n:Person)-[]->(m:Movie) with n.name as name, count(m) as count return name, count order by count desc limit 15 """
    person_statistics =  exec_cypher_query(person_graph_query_str)

    # store = TripleStore()
    # for record in movie_graph_pair:
    #     nodes.append(Node(id=record[0],size=15, label=record[0],image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_spiderman.png"))
    #     nodes.append(Node(id=record[2],size=15, label=record[2],image="http://marvel-force-chart.surge.sh/marvel_force_chart_img/top_captainmarvel.png"))
    #     edges.append(Edge(source=record[0],target=record[2],label=record[1],size=15,type="CURVE_SMOOTH"))

    # agraph(nodes,edges,config)
        # node1=record[2]
        # link=record[1]
        # node2=record[0]
        # store.add_triple(node1, link, node2)
        # agraph(list(node1+node2), list(link), config)

    # agraph(nodes,edges,config)
    # st.write("Nodes loaded: " + str(len(store.getNodes())))
    # st.success("Done")
    # agraph(list(store.getNodes()), (store.getEdges() ), config)

    st.subheader("Real-Time Recommendation")
    st.info("""Recommend Top 3 Movies to Person""")
    graph = graphviz.Digraph()
    for record in movie_graph_pair:
        graph.edge(record[0], record[2])
    st.graphviz_chart(graph,True)

    #---
    ndarray_results = np.array(person_statistics)
    dfx = pd.DataFrame(ndarray_results, columns=['Name','Count'])
    dfx['Count'] = dfx['Count'].astype(int)
    st.subheader("The most active person in the movie industry")
    ccm = alt.Chart(dfx,height=800,width=1600).mark_circle(size=500).encode(
        x=alt.X('Name', sort=None), y='Count',color='Name').interactive()
    st.altair_chart(ccm,theme="streamlit",use_container_width=True)

    #---
    ndarray_results = np.array(released_statistics)
    dfx = pd.DataFrame(ndarray_results, columns=['Year','Count'])
    dfx['Count'] = dfx['Count'].astype(int)
    st.subheader("Year to release movie")
    st.bar_chart(dfx,x='Year',y='Count')

    try:
        ndarray_results = np.array(get_highest_score_movie)
        dfxbar = pd.DataFrame(ndarray_results, columns=["Movie","Avg_Rate"])
        dfxbar['Avg_Rate'] = dfxbar['Avg_Rate'].astype(float)
        dfxbar['Avg_Rate']= dfxbar['Avg_Rate'].round().astype(int)
        st.subheader("The highest rated movies")

        st.write(alt.Chart(dfxbar,height=600,width=1600).mark_bar().encode(
            x=alt.X('Movie', sort=None),
            y='Avg_Rate'
        ))
    except ValueError as ve:
        st.warning("Empty in the query result")
    except IndexError as ve2:
        st.warning("Empty in query result")

    try:
        ndarray_results = (get_most_watched_movie)
        ndarray_results = np.array(get_most_watched_movie)
        dfx = pd.DataFrame(ndarray_results, columns=['Movie','Watch'])
        dfx['Watch'] = dfx['Watch'].astype(int)
        st.subheader("The most watched movies")
        # st.bar_chart(dfx,x="Movie")
        st.write(alt.Chart(dfx,height=600,width=1600).mark_bar().encode(
            x=alt.X('Movie', sort=None),
            y='Watch'
        ))
    except ValueError as ve:
        st.warning("Empty in the query result")
    except IndexError as ve2:
        st.warning("Empty in query result")

 ## INPUT MENU
if selected == "Input And Train to Recommend":
    movies =["Jerry Maguire", "Stand By Me", "As Good as It Gets", "What Dreams May Come", "Snow Falling on Cedars", "You've Got Mail", "Sleepless in Seattle", "The Matrix", "Joe Versus the Volcano", "When Harry Met Sally"]
    watchers =["Kelly McGillis", "Val Kilmer", "Anthony Edwards", "Tom Skerritt", "Meg Ryan", "Tony Scott", "Jim Cash", "Renee Zellweger", "Kelly Preston", "Jerry O'Connell"]
    raing =range(0,11)
    page_tile = "Watch & Rate Movie Selection"
    page_icon = ":person-video2"
    layout = "centered"
    st.title(page_tile +" "+ page_icon)

    years =range(datetime.today().year-15,datetime.today().year)
    months =list(calendar.month_name[1:])

    person_query_str = "MATCH (n:Person) with n limit 100 RETURN n.name"
    another_person_query_str = "MATCH (n:Person) with n order by n.name desc limit 100 RETURN n.name"
    movie_query_str = "MATCH (n:Movie) with n limit 50 RETURN n.title"

    movie_query_list = exec_cypher_query(movie_query_str)
    person_query_list = exec_cypher_query(person_query_str)
    another_person_query_list = exec_cypher_query(another_person_query_str)

    if person_query_list != None:
       person_query_list = list(chain.from_iterable(person_query_list))

    if another_person_query_list != None:
       another_person_query_list = list(chain.from_iterable(another_person_query_list))

    if movie_query_list != None:
       movie_query_list = list(chain.from_iterable(movie_query_list))

    st.header(f"Data Entry")

    with st.form("watch_movie_form",clear_on_submit=True):
        col1,col2,col3,col4 =st.columns(4)
        col1.selectbox("Select Year:", years,key="year")
        col2.selectbox("Select Month:", months,key="month")
        col3.selectbox("Select Movie:", movie_query_list,key="movie")
        col4.selectbox("Select Watcher:", person_query_list,key="watcher")

        "---"
        submitted_watch = st.form_submit_button("Save Watched Data")
        if submitted_watch:
            sub_period = str(st.session_state["year"])+"_"+str(st.session_state["month"])
            sub_watcher = str(st.session_state["watcher"])
            sub_movie = str(st.session_state["movie"])

            print ("movie:"+sub_movie +"||"+"watcher:"+sub_watcher +"||"+"period:"+sub_period)

            label1 = "Person"
            label1_prop_name = "name"
            label1_prop_property = sub_watcher
            label2 = "Movie"
            label2_prop_name = "title"
            label2_prop_property = sub_movie
            rel_type = "WATCHED"
            rel_prop_name = "period"
            rel_prop_property = sub_period
            exec_cypher_write(label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property)
            st.success("save successfully")

        "---"
    with st.form("rate_movie_form",clear_on_submit=True):
        col3,col4,col5 =st.columns(3)
        col3.selectbox("Select Movie:", movie_query_list,key="movie_review")
        col4.selectbox("Select Reviewer:", person_query_list,key="reviewer")
        col5.selectbox("Select Rating:", raing,key="rating")
        "---"
        submitted_rating = st.form_submit_button("Save Rating Data")
        if submitted_rating:
            sub_reviewer = str(st.session_state["reviewer"])
            sub_movie = str(st.session_state["movie_review"])
            sub_rating =str(st.session_state["rating"])
            print ("movie:"+sub_movie +"||"+"reviewer:"+sub_reviewer +"||"+"rating:"+sub_rating)

            label1 = "Person"
            label1_prop_name = "name"
            label1_prop_property = sub_reviewer
            label2 = "Movie"
            label2_prop_name = "title"
            label2_prop_property = sub_movie
            rel_type = "RATED"
            rel_prop_name = "score"
            rel_prop_property = sub_rating
            exec_cypher_write(label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property)
            st.success("save successfully")

        "---"
    with st.form("who_knows_who",clear_on_submit=True):
        col6,col7 =st.columns(2)
        col6.selectbox("Select User:", person_query_list,key="first_user")
        col7.selectbox("Select Another User:", another_person_query_list,key="second_user")
        "---"
        submitted_rating = st.form_submit_button("Save People Network Data")
        if submitted_rating:
            sub_first_person = str(st.session_state["first_user"])
            sub_second_person = str(st.session_state["second_user"])
            print ("first person:"+sub_first_person +"||"+"second person:"+sub_second_person )

            label1 = "Person"
            label1_prop_name = "name"
            label1_prop_property = sub_first_person
            label2 = "Person"
            label2_prop_name = "name"
            label2_prop_property = sub_second_person
            rel_type = "KNOWS"
            rel_prop_name = "familarity"
            rel_prop_property = '1'
            exec_cypher_write(label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property)
            st.success("save successfully")

    with st.form("recommendation_enrichment_form",clear_on_submit=True):
        enrich_list =["Genre"]
        col1,col2 =st.columns(2)
        col1.selectbox("Select an item to enrich movie knowledge graph:", enrich_list,key="enrich")
        submitted_enrich_movie_graph = st.form_submit_button("Enrich Movie Graph")
        if submitted_enrich_movie_graph:
            sub_enrich_item = str(st.session_state["enrich"])

            movie_query_str = "MATCH (n:Movie) with n limit 40 RETURN n.title"
            movie_query_list = exec_cypher_query(movie_query_str)

            myPrompt ="""Return genre map array for the movie list below, and if the movie has multiple genre, just return the first one """ + \
            """movies = """+ str(movie_query_list) + """ + \
            in a format `[movie1:genre1, movie2:genre2. ......]`, not using code
            """
            # print(myPrompt)
            openai.api_key = API_KEY
            model = LAN_MODEL
            response = openai.Completion.create(engine=model, prompt=myPrompt, max_tokens=1000)
            generated_text = response.choices[0].text

            mystrlist= (generated_text.split(","))
            # print(mystrlist)
            for record in mystrlist:
                str_merge = """
                MATCH (n:Movie) where n.title = \"""""" + record.split(":")[0].strip() + """\" Merge (m:Genre {genre:\""""+record.split(":")[1].strip() +"""\"}) on create set m.genre = \""""+ record.split(":")[1].strip() + """\" MERGE (n)-[:IN_GENRE]->(m);
                """
                exec_simple_cypher_write(str_merge.strip())

            print('Enrich the genre complete')
            st.success("Enrich the genre complete")

    "---"
    with st.form("train_form",clear_on_submit=True):
        submitted_train_recommendation = st.form_submit_button("Train Recommendation")
        if submitted_train_recommendation:
            recommendation_result=recommdation_engine()
            st.success("Training complete")
    "---"
    with st.form("predict_form",clear_on_submit=True):
        submitted_predicate_recommendation = st.form_submit_button("Train Prediction")
        if submitted_predicate_recommendation:
            predicate_result=prediction_engine()
            st.success("Prediction complete")

if selected == "Movie ChatBot":

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm Majic Jar, How may I help you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']

    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()

    # User input
    ## Function for taking user provided prompt as input

    ## Applying the user input box
    with input_container:
        user_input = get_text()

    ## Conditional display of AI generated responses as a function of user provided prompts
    with response_container:
        if user_input:
            # response = get_results(user_input)
            actual_response = str(get_neo4j_feedllm_results(user_input)).replace("[","").replace("]","")
            if actual_response =="":
                highest_score_movie_query ="""
                match (n:Person)-[r:RATED]-(m:Movie)
                with m.title as title, avg(toInteger(r.score)) as average_rate
                return title,average_rate order by average_rate desc limit 3
                """
                get_highest_score_movie = gdsrun(highest_score_movie_query)
                print(get_highest_score_movie)
                get_highest_score_movie_list = get_highest_score_movie['title'].tolist()

                if len(get_highest_score_movie_list)==0:
                    most_watched_movie_query ="""
                    MATCH p=(n:Person)-[r:WATCHED]->(m:Movie)
                    with m.title as title, size(collect(r)) as watched
                    return title,watched order by watched desc limit 3
                    """
                    get_most_watched_movie=gdsrun(most_watched_movie_query)
                    get_most_watched_movie_list = get_most_watched_movie['title'].tolist()
                    if len(get_most_watched_movie_list)==0:
                        response = "Do something for the movie industry and then come back to find some fun stuff"
                    else:
                         response = "Ah, magic jar suggest watching "+ str((get_most_watched_movie_list)).replace("[","").replace("]","")
                else:
                    response = "Ah, magic jar suggest watching "+ str((get_highest_score_movie_list)).replace("[","").replace("]","")
            else:
                response = "Ah, magic jar suggest watching " + str(get_neo4j_feedllm_results(user_input)).replace("[","").replace("]","")

            print("--"+response)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))

    colored_header(label='', description='', color_name='blue-30')
    st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
        border: none !important;
        font-family: "Source Sans Pro", sans-serif;
        color: rgba(49, 51, 63, 0.6);
        font-size: 0.9rem;
    }

    tr {
        border: none !important;
    }

    th {
        text-align: center;
        colspan: 3;
        border: none !important;
        color: #0F9D58;
    }

    th, td {
        padding: 2px;
        border: none !important;
    }
    </style>

    <table>
    <tr>
        <th colspan="3">Sample Questions to try out</th>
    </tr>
    <tr>
        <td>Can you recommend a movie for Keanu Reeves?</td>
        <td>Any prediction for Keanu Revees?</td>
    </tr>
    <tr>
        <td>Can you predict a movie for John Goodman?</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)
