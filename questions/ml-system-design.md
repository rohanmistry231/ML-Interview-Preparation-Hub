# ML System Design Questions

This file contains machine learning system design questions commonly asked in interviews at companies like **Google**, **Uber**, and **Amazon**. These questions assess your ability to **design scalable, production-ready ML systems**, covering functional and non-functional requirements, architecture, data pipelines, model training, evaluation, and deployment. They test your end-to-end understanding of building robust ML solutions.

Below are the questions with detailed answers, including step-by-step designs, key considerations, and practical insights for interviews.

---

## Table of Contents

1. [Build a real-time translation system](#1-build-a-real-time-translation-system)
2. [Build a real-time ETA model](#2-build-a-real-time-eta-model)
3. [Build a recommender system for product search](#3-build-a-recommender-system-for-product-search)

---

## 1. Build a real-time translation system

**Question**: [Google] Design a real-time translation system to translate text or speech across languages with low latency.

**Answer**:

A **real-time translation system** converts text or speech from one language to another instantly, requiring low latency, high accuracy, and scalability. Here’s a detailed design:

1. **Requirements**:
   - **Functional**:
     - Input: Text or speech in source language.
     - Output: Translated text or speech in target language.
     - Supported languages: e.g., 50+ languages.
   - **Non-Functional**:
     - Latency: <200ms for text, <1s for speech-to-speech.
     - Scalability: Handle millions of requests per second.
     - Accuracy: Comparable to state-of-the-art models (e.g., BLEU score >30).
     - Availability: 99.9% uptime.

2. **Architecture Overview**:
   - **Components**:
     - **Speech-to-Text (STT)**: Converts speech input to text (if speech-based).
     - **Text Translation Model**: Translates source text to target text.
     - **Text-to-Speech (TTS)**: Converts translated text to speech (if needed).
     - **Orchestrator**: Manages pipeline (STT → Translation → TTS).
     - **Caching Layer**: Stores frequent translations for speed.
     - **Load Balancer**: Distributes requests across servers.
     - **Monitoring**: Tracks latency, errors, and model drift.
   - **Flow**:
     - User sends input (text/speech) via API.
     - Load balancer routes to orchestrator.
     - Pipeline processes: STT (if speech), translate, TTS (if speech output).
     - Response returned to user.

3. **Data Preparation**:
   - **Dataset**: Use parallel corpora (e.g., OPUS, CommonCrawl) with millions of sentence pairs per language.
   - **Preprocessing**:
     - Tokenize text, normalize (e.g., lowercase, remove special characters).
     - Handle rare languages with transfer learning from high-resource languages.
   - **Features**: Token embeddings, language IDs, context (if conversational).

4. **Model Training**:
   - **Algorithm**: Transformer-based model (e.g., mBART, T5) for sequence-to-sequence translation.
   - **Why Transformer**: State-of-the-art for multilingual translation, handles long dependencies.
   - **Training**:
     - Pretrain on large corpora (e.g., Wikipedia dumps).
     - Fine-tune on domain-specific data (e.g., user queries).
     - Use mixed-precision training for efficiency.
   - **Optimization**: Quantization (e.g., INT8) and pruning to reduce latency.
   - **Hardware**: Train on GPUs/TPUs, deploy on CPUs/GPUs for inference.

5. **Model Evaluation**:
   - **Metrics**:
     - **BLEU**: Measures translation quality against reference translations.
     - **Latency**: End-to-end response time.
     - **User Satisfaction**: A/B test with human feedback.
   - **Offline**: Evaluate on held-out test set (e.g., WMT benchmarks).
   - **Online**: Monitor real-time metrics (e.g., error rate, latency).

6. **Model Productionization**:
   - **Serving**:
     - Use ONNX or TensorRT for optimized inference.
     - Deploy on Kubernetes with auto-scaling for traffic spikes.
   - **Caching**: Store frequent translations (e.g., “Hello” → “Hola”) in Redis.
   - **STT/TTS**: Integrate third-party APIs (e.g., Google STT, Amazon Polly) or custom models.
   - **API**: REST/gRPC endpoint (e.g., `POST /translate {text, source_lang, target_lang}`).
   - **Latency Optimization**:
     - Batch small requests during inference.
     - Use edge servers for low-latency delivery.
   - **Fault Tolerance**: Replicate services across regions, retry failed requests.

7. **Monitoring and Maintenance**:
   - **Metrics**: Latency, throughput, error rate, BLEU score drift.
   - **Drift**: Retrain if new slang or phrases emerge.
   - **Logging**: Store anonymized requests for analysis.
   - **A/B Testing**: Compare new models against baseline.

**Example**:
- Input: “Hello, how are you?” (English speech).
- Flow: STT → Text (“Hello, how are you?”) → Translation → Text (“Hola, ¿cómo estás?”) → TTS → Spanish speech.
- Latency: 500ms end-to-end, scalable to 1M users.

**Interview Tips**:
- Start with **requirements**: Clarify text vs. speech, latency goals.
- Sketch **architecture**: Draw pipeline (STT → Translation → TTS).
- Discuss **trade-offs**: Custom STT vs. third-party, model size vs. latency.
- Highlight **scalability**: Mention caching, auto-scaling, edge servers.

---

## 2. Build a real-time ETA model

**Question**: [Uber] Design a real-time ETA (Estimated Time of Arrival) model for a ride-sharing platform.

**Answer**:

A **real-time ETA model** predicts the time it takes for a driver to reach a passenger or complete a trip, requiring low latency, high accuracy, and robustness to dynamic conditions. Here’s the design:

1. **Requirements**:
   - **Functional**:
     - Input: Pickup/drop-off coordinates, trip details.
     - Output: ETA in minutes (e.g., 12.5 min).
   - **Non-Functional**:
     - Latency: <100ms per request.
     - Scalability: Handle millions of requests per minute.
     - Accuracy: Within ±2 minutes for 90% of predictions.
     - Availability: 99.99% uptime.

2. **Architecture Overview**:
   - **Components**:
     - **Data Ingestion**: Real-time feeds (GPS, traffic, weather).
     - **Feature Store**: Precomputed features (e.g., historical ETAs).
     - **ETA Model**: Predicts travel time.
     - **Routing Engine**: Computes optimal paths.
     - **Orchestrator**: Combines features, model, and routing.
     - **Caching Layer**: Stores recent ETAs for similar routes.
     - **Monitoring**: Tracks accuracy, latency, drift.
   - **Flow**:
     - User requests ETA via app.
     - Orchestrator pulls features, queries model and routing engine.
     - Response (ETA) returned to user.

3. **Data Preparation**:
   - **Dataset**:
     - Historical trips: `start/end coordinates`, `distance`, `duration`, `timestamp`.
     - Real-time: GPS traces, traffic (e.g., Google Maps API), weather.
     - Contextual: Road type, time of day, events (e.g., concerts).
   - **Preprocessing**:
     - Clean GPS noise (e.g., Kalman filter).
     - Aggregate traffic (e.g., avg. speed per road segment).
   - **Features**:
     - **Trip**: Distance, number of turns, road types.
     - **Dynamic**: Traffic speed, weather conditions.
     - **Temporal**: Hour, day, rush hour flags.
     - **Historical**: Avg. ETA for similar routes.

4. **Model Training**:
   - **Algorithm**: Gradient Boosted Trees (e.g., XGBoost) or Neural Networks.
   - **Why XGBoost**: Handles non-linear interactions, fast inference, robust to missing data.
   - **Training**:
     - Target: Actual trip duration (minutes).
     - Train on recent data (e.g., last 3 months) to capture trends.
     - Use weighted loss for outliers (e.g., heavy traffic).
   - **Optimization**: Feature selection to reduce inference time.

5. **Model Evaluation**:
   - **Metrics**:
     - **MAE**: Mean absolute error (target <2 min).
     - **Percentile Accuracy**: 90th percentile error.
     - **Latency**: Inference time.
   - **Offline**: Test on held-out trips.
   - **Online**: A/B test with drivers (e.g., ETA vs. actual).

6. **Model Productionization**:
   - **Serving**:
     - Deploy model on Kubernetes with gRPC for low-latency inference.
     - Use ONNX for cross-platform compatibility.
   - **Feature Store**: Store precomputed features (e.g., historical ETAs) in Redis.
   - **Routing Engine**: Integrate with OSRM or Google Maps for path planning.
   - **Caching**: Cache ETAs for frequent routes (e.g., airport to downtown).
   - **Latency Optimization**:
     - Precompute static features (e.g., road distances).
     - Parallelize model and routing queries.
   - **Fault Tolerance**: Fallback to historical averages if model fails.

7. **Monitoring and Maintenance**:
   - **Metrics**: MAE, latency, cache hit rate, request volume.
   - **Drift**: Retrain weekly for new traffic patterns.
   - **Alerts**: Trigger if MAE exceeds threshold.
   - **Logging**: Store predictions for analysis.

**Example**:
- Input: Trip from downtown to airport, 5 PM, rainy.
- Features: Distance (10 km), traffic (slow), historical ETA (15 min).
- Output: ETA = 18 min, MAE = 1.5 min.
- Scalability: Handles 10M daily requests.

**Interview Tips**:
- Clarify **scope**: “Is this driver-to-passenger or full trip ETA?”
- Draw **pipeline**: Ingestion → Features → Model → Routing.
- Discuss **real-time challenges**: “Traffic changes fast, so I’d use a feature store.”
- Highlight **accuracy vs. latency**: “Caching reduces latency but risks stale data.”

---

## 3. Build a recommender system for product search

**Question**: [Amazon] Design a recommender system to enhance product search results on an e-commerce platform.

**Answer**:

A **recommender system for product search** suggests relevant products based on user queries, improving search quality and conversion rates. It combines search relevance with personalization. Here’s the design:

1. **Requirements**:
   - **Functional**:
     - Input: User query (e.g., “wireless headphones”), user profile.
     - Output: Ranked list of products.
   - **Non-Functional**:
     - Latency: <200ms per query.
     - Scalability: Handle billions of searches daily.
     - Relevance: High click-through rate (CTR), conversion rate.
     - Availability: 99.9% uptime.

2. **Architecture Overview**:
   - **Components**:
     - **Query Processor**: Parses and normalizes queries.
     - **Search Index**: Retrieves candidate products.
     - **Recommender Model**: Reranks candidates with personalization.
     - **Feature Store**: Stores user/product features.
     - **Caching Layer**: Caches frequent query results.
     - **Orchestrator**: Combines search and recommendation.
     - **Monitoring**: Tracks CTR, latency, relevance.
   - **Flow**:
     - User enters query.
     - Query processor fetches candidates from search index.
     - Recommender reranks based on user profile.
     - Top products returned.

3. **Data Preparation**:
   - **Dataset**:
     - User data: `user_id`, `search_history`, `purchases`, `clicks`.
     - Product data: `product_id`, `title`, `category`, `price`, `ratings`.
     - Interaction data: Query-product pairs, clicks, purchases.
   - **Preprocessing**:
     - Tokenize queries, remove stop words.
     - Normalize product titles (e.g., stemming).
   - **Features**:
     - **Query**: Keywords, embeddings (e.g., BERT).
     - **User**: Purchase history, preferred categories, price range.
     - **Product**: Category, price, popularity, ratings.
     - **Context**: Time, device type.

4. **Model Training**:
   - **Algorithm**:
     - **Collaborative Filtering**: Matrix factorization (e.g., ALS) for user-item interactions.
     - **Content-Based**: Embeddings (e.g., product titles via BERT).
     - **Two-Tower Model**: Neural network with user and item towers for ranking.
   - **Why Two-Tower**: Balances personalization and relevance, scalable for inference.
   - **Training**:
     - Target: Click or purchase (binary classification/ranking).
     - Loss: Pairwise ranking (e.g., BPR) or pointwise (e.g., cross-entropy).
     - Train on recent interactions (e.g., last 6 months).

5. **Model Evaluation**:
   - **Metrics**:
     - **NDCG**: Measures ranking quality.
     - **CTR**: Clicks per impression.
     - **Conversion Rate**: Purchases per recommendation.
   - **Offline**: Evaluate on held-out interactions.
   - **Online**: A/B test with live users.

6. **Model Productionization**:
   - **Serving**:
     - Deploy on TensorFlow Serving or TorchServe.
     - Use Elasticsearch for search index, Redis for feature store.
   - **Caching**: Cache top queries (e.g., “iPhone”) and their results.
   - **Latency Optimization**:
     - Precompute embeddings for products.
     - Limit candidates (e.g., top 100 from search index).
   - **Scalability**:
     - Shard search index by category.
     - Auto-scale inference servers for peak traffic.
   - **Fault Tolerance**: Fallback to non-personalized search if model fails.

7. **Monitoring and Maintenance**:
   - **Metrics**: NDCG, CTR, latency, cache hit rate.
   - **Drift**: Retrain weekly for new products/queries.
   - **Cold Start**: Use content-based features for new users/products.
   - **Logging**: Store queries and clicks for analysis.

**Example**:
- Query: “wireless headphones.”
- Features: User prefers electronics, $50-$100 range.
- Output: Top 5 headphones, ranked by relevance + user history.
- Result: CTR improves by 15% vs. baseline search.

**Interview Tips**:
- Clarify **personalization**: “Should it prioritize user history or query match?”
- Draw **pipeline**: Query → Search → Rerank → Response.
- Discuss **cold start**: “For new users, I’d use content-based filtering.”
- Highlight **scalability**: “Sharding and caching handle billions of queries.”

---

## Notes

- **Comprehensiveness**: Answers cover end-to-end design, from requirements to monitoring.
- **Practicality**: Designs balance accuracy, latency, and scalability for real-world systems.
- **Clarity**: Explanations are structured for verbal delivery, with clear components and trade-offs.
- **Consistency**: Matches the style of `ml-coding.md`, `ml-theory.md`, `ml-algorithms.md`, and `applied-ml-cases.md`.

For deeper practice, explore [ML Coding](ml-coding.md) for implementation details or [Applied ML Cases](applied-ml-cases.md) for business-focused problems. 🚀

---

**Next Steps**: Solidify your prep with [Easy ML Questions](easy-ml.md) for fundamentals or dive into [Feature Engineering](feature-engineering.md) for data prep techniques! 🌟