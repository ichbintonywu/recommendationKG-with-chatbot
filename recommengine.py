from neo4j_handler import exec_simple_cypher_write,gdsrun
import pandas as pd

gds_project ="""
CALL gds.graph.project(
    'myMovie',
    ['Person', 'Movie'],
    'WATCHED'
);
"""
gds_written="""
CALL gds.nodeSimilarity.write('myMovie', {
    writeRelationshipType: 'SIMILAR',
    writeProperty: 'score'
})
YIELD nodesCompared, relationshipsWritten
"""
gds_remove_similar="""
MATCH p=()-[r:SIMILAR]->() delete r
"""
gds_dropprojection="""
call gds.graph.drop('myMovie')
"""
delete_previous_recommendation="""MATCH (n:Recommendation) detach delete n"""
merge_recommendation_person="""MATCH (n:Person) WHERE n.name = \""""
merge_recommendation_recommendation="""\" MERGE (r:Recommendation {movies:"""
merge_recommendation_recommendation2="""}) on create set r.movies = """
merge_recommendation_relationship =""" MERGE(n)-[:HAS_RECOMMENDATION]->(r)"""

def recommdation_engine():

    run_delete_previous_recommendation=gdsrun(delete_previous_recommendation)

    sort_by_score ="""
    match (n:Person)-[r:RATED]-(m:Movie) 
    with m.title as title, avg(toInteger(r.score)) as average_rate
    return title,average_rate order by average_rate desc limit 3
    """
    get_sort_by_score = gdsrun(sort_by_score)
    _order=[3,2,1]
    get_sort_by_score['score']=_order
    recommend_by_score=get_sort_by_score.filter(['title','score'])

    sort_by_watch ="""
    MATCH p=(n:Person)-[r:WATCHED]->(m:Movie) 
    with m.title as title, size(collect(r)) as watched
    return title,watched order by watched desc limit 3
    """
    get_sort_by_watch = gdsrun(sort_by_watch)
    get_sort_by_watch['score']=_order
    recommend_by_watch=get_sort_by_watch.filter(['title','score'])


    sort_by_category ="""
    match (o:Person)-[:WATCHED]->(m:Movie)
    -[:IN_GENRE]->(o2:Genre)<-[:IN_GENRE]-(p2:Movie)
    where (not (o)-[:WATCHED]->(p2)) and p2<>m
    with o.name as name, collect(p2.title) as titles
    return name,titles[0..3] as category
    """
    get_sort_by_category = gdsrun(sort_by_category)

    run_gds_project = gdsrun(gds_project)
    run_gds_delete_similar = gdsrun(gds_remove_similar)
    run_gds_written = gdsrun(gds_written)
    run_gds_dropprojection = gdsrun(gds_dropprojection)

    sort_by_similarity ="""
    MATCH p=(n:Person)-[r:SIMILAR]->(m:Person)--(movie:Movie) where id(n)<id(m)
    with  n.name as name, (movie.title) as title, (r.score) as score
    with name,title order by score desc
    with name,apoc.coll.toSet(collect(title))[0..3] as similarity where size(similarity)>2
    return name,similarity
    """
    get_sort_by_similarity = gdsrun(sort_by_similarity)


    df_merged=pd.merge(get_sort_by_category,get_sort_by_similarity, on='name',how='outer')

    recommend_by_score_list=recommend_by_score['title'].tolist()
    recommend_by_watch_list=get_sort_by_watch['title'].tolist()

    df_merged['score']=df_merged.apply(lambda row:recommend_by_score_list,axis=1)
    df_merged['watch']=df_merged.apply(lambda row:recommend_by_watch_list,axis=1)

    print(df_merged)
    df_merged.dropna(inplace=True)
    for index, row in df_merged.iterrows():
        name = row['name']
        category_df = pd.DataFrame(row['category'])
        category_df.rename(columns={0: 'category'}, inplace=True)
        category_df['WeightedValue']= [3,2,1]

        if hasattr(row['similarity'],"__len__"): 
            similarity_df = pd.DataFrame(row['similarity'])
            similarity_df.rename(columns={0: 'similarity'}, inplace=True)
            similarity_df['WeightedValue']= [3,2,1] 

        score_df = pd.DataFrame(row['score'])
        score_df.rename(columns={0: 'score'}, inplace=True)
        score_df['WeightedValue']= [3,2,1] 
        
        watch_df = pd.DataFrame(row['watch'])
        watch_df.rename(columns={0: 'watch'}, inplace=True)
        watch_df['WeightedValue']= [3,2,1] 
        

        # Assign priorities to the dataframes
        category_priority = 0.1
        score_priority = 0.3
        similarity_priority = 0.4
        swatch_priority = 0.2

        category_df['WeightedValue'] = category_df['WeightedValue'] * category_priority
        score_df['WeightedValue'] = score_df['WeightedValue'] * score_priority
        watch_df['WeightedValue'] = watch_df['WeightedValue'] * swatch_priority
        similarity_df['WeightedValue'] = similarity_df['WeightedValue'] * similarity_priority
        
        # Concatenate the dataframes
        result = pd.concat([score_df, category_df, watch_df, similarity_df])

        # Sort by weighted value in descending order
        result.sort_values(by='WeightedValue', ascending=False, inplace=True)

        # Reset index if needed
        result.reset_index(drop=True, inplace=True)

        df_result=result
        # Print the result
    #     result1=result.values.flatten().tolist()
        merged_column = df_result['score'].combine_first(df_result['category']).combine_first(df_result['watch']).combine_first(df_result['similarity'])
        
        merged_list = list(set(merged_column.tolist()))
        # Output the result
        # print(name,merged_list[:3])
        merge_str= merge_recommendation_person + name +""""""+merge_recommendation_recommendation+str(merged_list[:3])+ merge_recommendation_recommendation2+str(merged_list[:3])+merge_recommendation_relationship
        # print(merge_str)
        exec_simple_cypher_write(merge_str)
    # build relation from user name to Recommendation with the list 


def prediction_engine():
    pipeline_handling_str = """
        MATCH path=(n:Person)-[r:SIMILAR]->(m) -[p:SIMILAR]->(n) where id(m)>id(n) delete p;
        CALL gds.graph.project(
        'persons',
        'Person',
        {
            SIMILAR: {
            orientation: 'UNDIRECTED',
            properties: 'score'
            }
        }
        );
        CALL gds.beta.pipeline.linkPrediction.create('pipe');
        CALL gds.beta.pipeline.linkPrediction.addNodeProperty('pipe', 'fastRP', {
        mutateProperty: 'fastrp-embedding',
        embeddingDimension: 36,
        randomSeed: 42
        });
        CALL gds.beta.pipeline.linkPrediction.addFeature('pipe', 'hadamard', {
        nodeProperties: ['fastrp-embedding']
        }) YIELD featureSteps;
        CALL gds.beta.pipeline.linkPrediction.addRandomForest('pipe', {numberOfDecisionTrees: 10})
        YIELD parameterSpace;
        CALL gds.alpha.pipeline.linkPrediction.addMLP('pipe', {hiddenLayerSizes: [4, 2], penalty: 1, patience: 2});
        CALL gds.beta.pipeline.linkPrediction.configureSplit('pipe', {
        testFraction: 0.25,
        trainFraction: 0.8,
        validationFolds: 5
        });
        CALL gds.beta.pipeline.linkPrediction.train('persons', {
        pipeline: 'pipe',
        modelName: 'lp-pipeline-model-filtered',
        metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
        sourceNodeLabel: 'Person',
        targetNodeLabel: 'Person',
        targetRelationshipType: 'SIMILAR',
        randomSeed: 12
        }) YIELD modelInfo, modelSelectionStats
        RETURN
        modelInfo.bestParameters AS winningModel,
        modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
        modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
        modelInfo.metrics.AUCPR.test AS testScore,
        [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores;
        CALL gds.beta.pipeline.linkPrediction.predict.stream('persons', {
        modelName: 'lp-pipeline-model-filtered',
        topN: 5,
        threshold: 0.35
        })
        YIELD node1, node2, probability
        with gds.util.asNode(node1) AS person1, gds.util.asNode(node2) AS person2, probability
        ORDER BY probability DESC, person1
        merge (person1)-[r:PREDICATED_SIMILAR]->(person2) set r.probability=probability;
        call gds.beta.model.drop('lp-pipeline-model-filtered');
        CALL gds.beta.pipeline.drop('pipe') YIELD pipelineName, pipelineType;
        call gds.graph.drop('persons')
        """
    single_pipeline_run_list = pipeline_handling_str.split(";")
    predict_merge_str = """
    MATCH (p1:Person)-[r:PREDICATED_SIMILAR]-(p2:Person)-[:WATCHED]-(m:Movie) with p1, collect(m.title) as predict merge (rec:Recommendation {movies:predict}) merge(p1)-[:PREDICATED_RECOMMEND]->(rec)
    """
    for cypher_stat in single_pipeline_run_list:
        # print(cypher_stat)
        # test merge match(p:Person{name:'Matthew Fox'}) match (m:Movie {title:'Speed Racer'} ) merge (p)-[r:WATCHED]->(m) on create set r.period='2021_January'
        ## test prediction MATCH p=(p1:Person)-[r:PREDICATED_SIMILAR]-(p2:Person)-[:WATCHED]-(m:Movie) return p1.name,m.title
        ### merge new rel MATCH (p1:Person)-[r:PREDICATED_SIMILAR]-(p2:Person)-[:WATCHED]-(m:Movie) with p1, collect(m.title) as predict merge (rec:Recommendation {movies:predict}) merge(p1)-[:PREDICATED_RECOMMEND]->(rec)
        gdsrun(cypher_stat)
        gdsrun(predict_merge_str)