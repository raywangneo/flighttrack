export type WeatherSource = 'forecast' | 'historical_average';
export type RiskLevel = 'low' | 'medium' | 'high';

export interface PredictRequest {
  airline: string;
  origin: string;
  dest: string;
  scheduled_departure: string; // ISO8601 local time at origin, no offset
}

export interface PredictResponse {
  delay_probability: number;
  delayed_prediction: boolean;
  risk_level: RiskLevel;
  weather_source: WeatherSource;
  model_version: string;
  caveats: string[];
}
