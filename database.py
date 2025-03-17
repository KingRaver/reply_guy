def check_content_similarity_with_timeframe(self, content: str, hours: int = 1, timeframe: str = None) -> bool:
        """
        Check if similar content was posted within a specified timeframe
        Can filter by prediction timeframe
        """
        conn, cursor = self._get_connection()
        try:
            query = """
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            
            params = [hours]
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            cursor.execute(query, params)
            recent_posts = [row['content'] for row in cursor.fetchall()]
            
            # Split content into main text and hashtags
            content_main = content.split("\n\n#")[0].lower() if "\n\n#" in content else content.lower()
            
            for post in recent_posts:
                post_main = post.split("\n\n#")[0].lower() if "\n\n#" in post else post.lower()
                
                # Calculate similarity based on word overlap
                content_words = set(content_main.split())
                post_words = set(post_main.split())
                
                if content_words and post_words:
                    overlap = len(content_words.intersection(post_words))
                    similarity = overlap / max(len(content_words), len(post_words))
                    
                    # Consider similar if 70% or more words overlap
                    if similarity > 0.7:
                        return True
            
            return False
        except Exception as e:
            logger.log_error("Check Content Similarity With Timeframe", str(e))
            return False

    #########################
    # PREDICTION METHODS
    #########################
    
    def store_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> int:
        """
        Store a prediction in the database
        Returns the ID of the inserted prediction
        """
        conn, cursor = self._get_connection()
        prediction_id = None
        
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            model_weights = json.dumps(prediction_data.get("model_weights", {}))
            model_inputs = json.dumps(prediction_data.get("inputs", {}))
            
            # Calculate expiration time based on timeframe
            if timeframe == "1h":
                expiration_time = datetime.now() + timedelta(hours=1)
            elif timeframe == "24h":
                expiration_time = datetime.now() + timedelta(hours=24)
            elif timeframe == "7d":
                expiration_time = datetime.now() + timedelta(days=7)
            else:
                expiration_time = datetime.now() + timedelta(hours=1)  # Default to 1h
                
            cursor.execute("""
                INSERT INTO price_predictions (
                    timestamp, token, timeframe, prediction_type,
                    prediction_value, confidence_level, lower_bound, upper_bound,
                    prediction_rationale, method_weights, model_inputs, technical_signals,
                    expiration_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    "price",
                    prediction["price"],
                    prediction["confidence"],
                    prediction["lower_bound"],
                    prediction["upper_bound"],
                    rationale,
                    model_weights,
                    model_inputs,
                    key_factors,
                    expiration_time
                ))
                
            conn.commit()
            prediction_id = cursor.lastrowid
            logger.logger.debug(f"Stored {timeframe} prediction for {token} with ID {prediction_id}")
            
            # Also store in specialized tables based on the prediction models used
            
            # Store Claude prediction if it was used
            if prediction_data.get("model_weights", {}).get("claude_enhanced", 0) > 0:
                self._store_claude_prediction(token, prediction_data, timeframe)
                
            # Store technical analysis if available
            if "inputs" in prediction_data and "technical_analysis" in prediction_data["inputs"]:
                self._store_technical_indicators(token, prediction_data["inputs"]["technical_analysis"], timeframe)
                
            # Store statistical forecast if available
            if "inputs" in prediction_data and "statistical_forecast" in prediction_data["inputs"]:
                self._store_statistical_forecast(token, prediction_data["inputs"]["statistical_forecast"], timeframe)
                
            # Store ML forecast if available
            if "inputs" in prediction_data and "ml_forecast" in prediction_data["inputs"]:
                self._store_ml_forecast(token, prediction_data["inputs"]["ml_forecast"], timeframe)
                
            # Update timeframe metrics
            self._update_timeframe_metrics(token, timeframe, prediction_data)
            
        except Exception as e:
            logger.log_error(f"Store Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
            
        return prediction_id
    
    def _store_technical_indicators(self, token: str, technical_analysis: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store technical indicator data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract indicator values
            overall_trend = technical_analysis.get("overall_trend", "neutral")
            trend_strength = technical_analysis.get("trend_strength", 50)
            signals = technical_analysis.get("signals", {})
            
            # Extract individual indicators if available
            indicators = technical_analysis.get("indicators", {})
            
            # Get RSI
            rsi = indicators.get("rsi", None)
            
            # Get MACD
            macd = indicators.get("macd", {})
            macd_line = macd.get("macd_line", None)
            signal_line = macd.get("signal_line", None)
            histogram = macd.get("histogram", None)
            
            # Get Bollinger Bands
            bb = indicators.get("bollinger_bands", {})
            bb_upper = bb.get("upper", None)
            bb_middle = bb.get("middle", None)
            bb_lower = bb.get("lower", None)
            
            # Get Stochastic
            stoch = indicators.get("stochastic", {})
            stoch_k = stoch.get("k", None)
            stoch_d = stoch.get("d", None)
            
            # Get OBV
            obv = indicators.get("obv", None)
            
            # Get ADX
            adx = indicators.get("adx", None)
            
            # Get additional timeframe-specific indicators
            ichimoku_data = json.dumps(indicators.get("ichimoku", {}))
            pivot_points = json.dumps(indicators.get("pivot_points", {}))
            
            # Get volatility
            volatility = technical_analysis.get("volatility", None)
            
            # Store in database
            cursor.execute("""
                INSERT INTO technical_indicators (
                    timestamp, token, timeframe, rsi, macd_line, macd_signal, 
                    macd_histogram, bb_upper, bb_middle, bb_lower,
                    stoch_k, stoch_d, obv, adx, ichimoku_data, pivot_points,
                    overall_trend, trend_strength, volatility, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                rsi,
                macd_line,
                signal_line,
                histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                stoch_k,
                stoch_d,
                obv,
                adx,
                ichimoku_data,
                pivot_points,
                overall_trend,
                trend_strength,
                volatility,
                json.dumps(technical_analysis)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Technical Indicators - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_statistical_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store statistical forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type from model_info if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "ARIMA")
            
            # Extract model parameters if available
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO statistical_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    model_parameters, input_data_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                model_parameters,
                "{}"   # Input data summary (empty for now)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Statistical Forecast - {token} ({timeframe})", str(e))
            conn.rollback()

    def _store_ml_forecast(self, token: str, forecast_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store machine learning forecast data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract forecast and confidence intervals
            forecast_value = forecast_data.get("prediction", 0)
            confidence = forecast_data.get("confidence", [0, 0])
            
            # Get model type and parameters if available
            model_info = forecast_data.get("model_info", {})
            model_type = model_info.get("method", "RandomForest")
            
            # Extract feature importance if available
            feature_importance = json.dumps(forecast_data.get("feature_importance", {}))
            
            # Store model parameters
            model_parameters = json.dumps(model_info)
            
            # Store in database
            cursor.execute("""
                INSERT INTO ml_forecasts (
                    timestamp, token, timeframe, model_type,
                    forecast_value, confidence_80_lower, confidence_80_upper,
                    confidence_95_lower, confidence_95_upper, 
                    feature_importance, model_parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                model_type,
                forecast_value,
                confidence[0],  # 80% confidence lower
                confidence[1],  # 80% confidence upper
                confidence[0] * 0.9,  # Approximate 95% confidence lower
                confidence[1] * 1.1,  # Approximate 95% confidence upper
                feature_importance,
                model_parameters
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store ML Forecast - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _store_claude_prediction(self, token: str, prediction_data: Dict[str, Any], timeframe: str = "1h") -> None:
        """Store Claude AI prediction data with timeframe support"""
        conn, cursor = self._get_connection()
        try:
            # Extract prediction details
            prediction = prediction_data.get("prediction", {})
            rationale = prediction_data.get("rationale", "")
            sentiment = prediction_data.get("sentiment", "NEUTRAL")
            key_factors = json.dumps(prediction_data.get("key_factors", []))
            
            # Default Claude model
            claude_model = "claude-3-7-sonnet-20250219"
            
            # Store inputs if available
            input_data = json.dumps(prediction_data.get("inputs", {}))
            
            # Store in database
            cursor.execute("""
                INSERT INTO claude_predictions (
                    timestamp, token, timeframe, claude_model,
                    prediction_value, confidence_level, sentiment,
                    rationale, key_factors, input_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                token,
                timeframe,
                claude_model,
                prediction.get("price", 0),
                prediction.get("confidence", 70),
                sentiment,
                rationale,
                key_factors,
                input_data
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Store Claude Prediction - {token} ({timeframe})", str(e))
            conn.rollback()
    
    def _update_timeframe_metrics(self, token: str, timeframe: str, prediction_data: Dict[str, Any]) -> None:
        """Update timeframe metrics based on new prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get current metrics for this token and timeframe
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            # Get prediction performance
            performance = self.get_prediction_performance(token=token, timeframe=timeframe)
            
            if performance:
                avg_accuracy = performance[0]["accuracy_rate"]
                total_count = performance[0]["total_predictions"]
                correct_count = performance[0]["correct_predictions"]
            else:
                avg_accuracy = 0
                total_count = 0
                correct_count = 0
            
            # Extract model weights
            model_weights = prediction_data.get("model_weights", {})
            
            # Determine best model
            if model_weights:
                best_model = max(model_weights.items(), key=lambda x: x[1])[0]
            else:
                best_model = "unknown"
            
            if metrics:
                # Update existing metrics
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        model_weights = ?,
                        best_model = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Insert new metrics
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    avg_accuracy,
                    total_count,
                    correct_count,
                    json.dumps(model_weights),
                    best_model,
                    datetime.now()
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_active_predictions(self, token: str = None, timeframe: str = None) -> List[Dict[str, Any]]:
        """
        Get all active (non-expired) predictions
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT * FROM price_predictions
                WHERE expiration_time > datetime('now')
            """
            params = []
            
            if token:
                query += " AND token = ?"
                params.append(token)
                
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Active Predictions", str(e))
            return []

    def get_expired_unevaluated_predictions(self, timeframe: str = None) -> List[Dict[str, Any]]:
        """
        Get all expired predictions that haven't been evaluated yet
        Can filter by timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = """
                SELECT p.* FROM price_predictions p
                LEFT JOIN prediction_outcomes o ON p.id = o.prediction_id
                WHERE p.expiration_time <= datetime('now')
                AND o.id IS NULL
            """
            
            params = []
            
            if timeframe:
                query += " AND p.timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY p.timeframe ASC, p.expiration_time ASC"
            
            cursor.execute(query, params)
            
            predictions = [dict(row) for row in cursor.fetchall()]
            
            # Parse JSON fields
            for prediction in predictions:
                prediction["method_weights"] = json.loads(prediction["method_weights"]) if prediction["method_weights"] else {}
                prediction["model_inputs"] = json.loads(prediction["model_inputs"]) if prediction["model_inputs"] else {}
                prediction["technical_signals"] = json.loads(prediction["technical_signals"]) if prediction["technical_signals"] else []
                
            return predictions
            
        except Exception as e:
            logger.log_error("Get Expired Unevaluated Predictions", str(e))
            return []

    def record_prediction_outcome(self, prediction_id: int, actual_price: float) -> bool:
        """Record the outcome of a prediction"""
        conn, cursor = self._get_connection()
        
        try:
            # Get the prediction details
            prediction = self.get_prediction_by_id(prediction_id)
            if not prediction:
                return False
                
            # Calculate accuracy metrics
            prediction_value = prediction["prediction_value"]
            lower_bound = prediction["lower_bound"]
            upper_bound = prediction["upper_bound"]
            timeframe = prediction["timeframe"]
            
            # Percentage accuracy (how close the prediction was)
            price_diff = abs(actual_price - prediction_value)
            accuracy_percentage = (1 - (price_diff / prediction_value)) * 100 if prediction_value > 0 else 0
            
            # Whether the actual price fell within the predicted range
            was_correct = lower_bound <= actual_price <= upper_bound
            
            # Deviation from prediction (for tracking bias)
            deviation = ((actual_price / prediction_value) - 1) * 100 if prediction_value > 0 else 0
            
            # Get market conditions at evaluation time
            market_data = self.get_recent_market_data(prediction["token"], 1)
            market_conditions = json.dumps({
                "evaluation_time": datetime.now().isoformat(),
                "token": prediction["token"],
                "market_data": market_data[:1] if market_data else []
            })
            
            # Store the outcome
            cursor.execute("""
                INSERT INTO prediction_outcomes (
                    prediction_id, actual_outcome, accuracy_percentage,
                    was_correct, evaluation_time, deviation_from_prediction,
                    market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                actual_price,
                accuracy_percentage,
                1 if was_correct else 0,
                datetime.now(),
                deviation,
                market_conditions
            ))
            
            # Update the performance summary
            token = prediction["token"]
            prediction_type = prediction["prediction_type"]
            
            self._update_prediction_performance(token, timeframe, prediction_type, was_correct, abs(deviation))
            
            # Update timeframe metrics
            self._update_timeframe_outcome_metrics(token, timeframe, was_correct, accuracy_percentage)
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.log_error(f"Record Prediction Outcome - {prediction_id}", str(e))
            conn.rollback()
            return False

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get a prediction by its ID"""
        conn, cursor = self._get_connection()
        
        try:
            cursor.execute("""
                SELECT * FROM price_predictions
                WHERE id = ?
            """, (prediction_id,))
            
            prediction = cursor.fetchone()
            if not prediction:
                return None
                
            # Convert to dict and parse JSON fields
            result = dict(prediction)
            result["method_weights"] = json.loads(result["method_weights"]) if result["method_weights"] else {}
            result["model_inputs"] = json.loads(result["model_inputs"]) if result["model_inputs"] else {}
            result["technical_signals"] = json.loads(result["technical_signals"]) if result["technical_signals"] else []
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Prediction By ID - {prediction_id}", str(e))
            return None

    def _update_prediction_performance(self, token: str, timeframe: str, prediction_type: str, was_correct: bool, deviation: float) -> None:
        """Update prediction performance summary"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if performance record exists
            cursor.execute("""
                SELECT * FROM prediction_performance
                WHERE token = ? AND timeframe = ? AND prediction_type = ?
            """, (token, timeframe, prediction_type))
            
            performance = cursor.fetchone()
            
            if performance:
                # Update existing record
                performance_dict = dict(performance)
                total_predictions = performance_dict["total_predictions"] + 1
                correct_predictions = performance_dict["correct_predictions"] + (1 if was_correct else 0)
                accuracy_rate = (correct_predictions / total_predictions) * 100
                
                # Update average deviation (weighted average)
                avg_deviation = (performance_dict["avg_deviation"] * performance_dict["total_predictions"] + deviation) / total_predictions
                
                cursor.execute("""
                    UPDATE prediction_performance
                    SET total_predictions = ?,
                        correct_predictions = ?,
                        accuracy_rate = ?,
                        avg_deviation = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (
                    total_predictions,
                    correct_predictions,
                    accuracy_rate,
                    avg_deviation,
                    datetime.now(),
                    performance_dict["id"]
                ))
                
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO prediction_performance (
                        token, timeframe, prediction_type, total_predictions,
                        correct_predictions, accuracy_rate, avg_deviation, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token,
                    timeframe,
                    prediction_type,
                    1,
                    1 if was_correct else 0,
                    100 if was_correct else 0,
                    deviation,
                    datetime.now()
                ))
                
        except Exception as e:
            logger.log_error(f"Update Prediction Performance - {token}", str(e))
            raise

    def _update_timeframe_outcome_metrics(self, token: str, timeframe: str, was_correct: bool, accuracy_percentage: float) -> None:
        """Update timeframe metrics with outcome data"""
        conn, cursor = self._get_connection()
        
        try:
            # Check if metrics record exists
            cursor.execute("""
                SELECT * FROM timeframe_metrics
                WHERE token = ? AND timeframe = ?
            """, (token, timeframe))
            
            metrics = cursor.fetchone()
            
            if metrics:
                # Update existing metrics
                metrics_dict = dict(metrics)
                total_count = metrics_dict["total_count"] + 1
                correct_count = metrics_dict["correct_count"] + (1 if was_correct else 0)
                
                # Recalculate average accuracy with new data point
                # Use weighted average based on number of predictions
                old_weight = (total_count - 1) / total_count
                new_weight = 1 / total_count
                avg_accuracy = (metrics_dict["avg_accuracy"] * old_weight) + (accuracy_percentage * new_weight)
                
                cursor.execute("""
                    UPDATE timeframe_metrics
                    SET avg_accuracy = ?,
                        total_count = ?,
                        correct_count = ?,
                        last_updated = ?
                    WHERE token = ? AND timeframe = ?
                """, (
                    avg_accuracy,
                    total_count,
                    correct_count,
                    datetime.now(),
                    token,
                    timeframe
                ))
            else:
                # Should not happen normally, but create metrics if missing
                cursor.execute("""
                    INSERT INTO timeframe_metrics (
                        timestamp, token, timeframe, avg_accuracy,
                        total_count, correct_count, model_weights,
                        best_model, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    token,
                    timeframe,
                    accuracy_percentage,
                    1,
                    1 if was_correct else 0,
                    "{}",
                    "unknown",
                    datetime.now()
                ))
                
            conn.commit()
            
        except Exception as e:
            logger.log_error(f"Update Timeframe Outcome Metrics - {token} ({timeframe})", str(e))
            conn.rollback()

    def get_prediction_performance(self, token: str = None, timeframe: str = None) -> List[Dict[str, Any]]:
        """
        Get prediction performance statistics
        Can filter by token and/or timeframe
        """
        conn, cursor = self._get_connection()
        
        try:
            query = "SELECT * FROM prediction_performance"
            params = []
            
            if token or timeframe:
                query += " WHERE "
                
            if token:
                query += "token = ?"
                params.append(token)
                
            if token and timeframe:
                query += " AND "
                
            if timeframe:
                query += "timeframe = ?"
                params.append(timeframe)
                
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.log_error("Get Prediction Performance", str(e))
            return []

    #########################
    # DATABASE MAINTENANCE METHODS
    #########################
            
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data to prevent database bloat
        Returns count of deleted records by table
        """
        conn, cursor = self._get_connection()
        
        tables_to_clean = [
            "market_data",
            "posted_content",
            "mood_history",
            "smart_money_indicators",
            "token_market_comparison",
            "token_correlations",
            "generic_json_data",
            "technical_indicators",
            "statistical_forecasts",
            "ml_forecasts",
            "claude_predictions",
            "replied_tweets"  # Include replied_tweets in cleanup
        ]
        
        deleted_counts = {}
        
        try:
            for table in tables_to_clean:
                # Keep prediction-related tables longer
                retention_days = days_to_keep * 2 if table in ["price_predictions", "prediction_outcomes"] else days_to_keep
                
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-' || ? || ' days')
                """, (retention_days,))
                
                deleted_counts[table] = cursor.rowcount
                
            # Special handling for evaluated predictions
            cursor.execute("""
                DELETE FROM price_predictions
                WHERE id IN (
                    SELECT p.id
                    FROM price_predictions p
                    JOIN prediction_outcomes o ON p.id = o.prediction_id
                    WHERE p.timestamp < datetime('now', '-' || ? || ' days')
                )
            """, (days_to_keep * 2,))
            
            deleted_counts["price_predictions"] = cursor.rowcount
            
            conn.commit()
            logger.logger.info(f"Database cleanup completed: {deleted_counts}")
            
            return deleted_counts
            
        except Exception as e:
            logger.log_error("Database Cleanup", str(e))
            conn.rollback()
            return {}
            
    def optimize_database(self) -> bool:
        """
        Optimize database performance by running VACUUM and ANALYZE
        """
        conn, cursor = self._get_connection()
        
        try:
            # Backup current connection settings
            old_isolation_level = conn.isolation_level
            
            # Set isolation level to None for VACUUM
            conn.isolation_level = None
            
            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            # Restore original isolation level
            conn.isolation_level = old_isolation_level
            
            logger.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.log_error("Database Optimization", str(e))
            return False
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including table sizes and row counts
        """
        conn, cursor = self._get_connection()
        
        try:
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row["name"] for row in cursor.fetchall()]
            
            stats = {
                "tables": {},
                "total_rows": 0,
                "last_optimized": None
            }
            
            # Get row count for each table
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                row_count = cursor.fetchone()["count"]
                
                # Get most recent timestamp if available
                try:
                    cursor.execute(f"SELECT MAX(timestamp) as last_update FROM {table}")
                    last_update = cursor.fetchone()["last_update"]
                except:
                    last_update = None
                
                stats["tables"][table] = {
                    "rows": row_count,
                    "last_update": last_update
                }
                
                stats["total_rows"] += row_count
                
            # Get database size (approximate)
            stats["database_size_kb"] = os.path.getsize(self.db_path) / 1024
            
            # Get last VACUUM time (if available in generic_json_data)
            cursor.execute("""
                SELECT timestamp FROM generic_json_data
                WHERE data_type = 'database_maintenance'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            last_maintenance = cursor.fetchone()
            if last_maintenance:
                stats["last_optimized"] = last_maintenance["timestamp"]
                
            return stats
            
        except Exception as e:
            logger.log_error("Get Database Stats", str(e))
            return {"error": str(e)}

    def close(self):
        """Close database connection"""
        if hasattr(self, 'local') and hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            self.local.conn = None
            self.local.cursor = None
