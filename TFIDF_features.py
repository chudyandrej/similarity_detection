from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from evaluation import AuthorityEvaluator, triplet_generator


if __name__ == '__main__':
    output_path = f"{os.environ['PYTHONPATH'].split(':')[0]}/outcome/baseline/TFIDF"
    os.makedirs(output_path, exist_ok=True)

    ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean", results_file=output_path)
    cvut_profiles = ev.cvut_profiles

    doc_profiles = []
    for profile in cvut_profiles:
        document = ""
        for val in profile.quantiles:
            document += " " + str(val)
        doc_profiles.append(document[1:])

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5))
    embedding_vectors = vectorizer.fit_transform(doc_profiles)
    embedding_vectors = embedding_vectors.todense()
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")
    print(embedding_vectors.shape)
    ev.evaluate_embeddings(cvut_profiles, embedding_vectors)



