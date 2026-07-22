import DateTimePicker from '@react-native-community/datetimepicker';
import { Picker } from '@react-native-picker/picker';
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { getAirlines, getAirports, predictDelay } from '../api/client';
import type { PredictResponse } from '../api/types';

const RISK_COLORS: Record<PredictResponse['risk_level'], string> = {
  low: '#2e7d32',
  medium: '#f9a825',
  high: '#c62828',
};

function toLocalIso(date: Date): string {
  // Strip timezone offset — the backend expects local time at the origin
  // airport with no offset, not the phone's own UTC-adjusted timestamp.
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}:00`;
}

export default function PredictScreen() {
  const [airlines, setAirlines] = useState<string[]>([]);
  const [airports, setAirports] = useState<string[]>([]);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [optionsError, setOptionsError] = useState<string | null>(null);

  const [airline, setAirline] = useState('');
  const [origin, setOrigin] = useState('');
  const [dest, setDest] = useState('');
  const [departure, setDeparture] = useState(new Date());
  const [showPicker, setShowPicker] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [airlineList, airportList] = await Promise.all([getAirlines(), getAirports()]);
        setAirlines(airlineList);
        setAirports(airportList);
        setAirline(airlineList[0] ?? '');
        setOrigin(airportList[0] ?? '');
        setDest(airportList[1] ?? airportList[0] ?? '');
      } catch (e) {
        setOptionsError(
          e instanceof Error
            ? `${e.message} — the backend may be waking up from sleep, try again in a moment.`
            : 'Failed to load airlines/airports.',
        );
      } finally {
        setLoadingOptions(false);
      }
    })();
  }, []);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictDelay({
        airline,
        origin,
        dest,
        scheduled_departure: toLocalIso(departure),
      });
      setResult(res);
    } catch (e) {
      setError(
        e instanceof Error
          ? `${e.message} — the backend may be waking up from sleep, try again in a moment.`
          : 'Prediction failed.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  if (loadingOptions) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
        <Text style={styles.hint}>Waking up the server…</Text>
      </View>
    );
  }

  if (optionsError) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{optionsError}</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>FlightTrack</Text>
      <Text style={styles.subtitle}>Check your flight&apos;s delay risk</Text>

      <Text style={styles.label}>Airline</Text>
      <View style={styles.pickerWrapper}>
        <Picker selectedValue={airline} onValueChange={setAirline}>
          {airlines.map((a) => (
            <Picker.Item key={a} label={a} value={a} />
          ))}
        </Picker>
      </View>

      <Text style={styles.label}>Origin</Text>
      <View style={styles.pickerWrapper}>
        <Picker selectedValue={origin} onValueChange={setOrigin}>
          {airports.map((a) => (
            <Picker.Item key={a} label={a} value={a} />
          ))}
        </Picker>
      </View>

      <Text style={styles.label}>Destination</Text>
      <View style={styles.pickerWrapper}>
        <Picker selectedValue={dest} onValueChange={setDest}>
          {airports.map((a) => (
            <Picker.Item key={a} label={a} value={a} />
          ))}
        </Picker>
      </View>

      <Text style={styles.label}>Scheduled departure (local time)</Text>
      <TouchableOpacity style={styles.dateButton} onPress={() => setShowPicker(true)}>
        <Text>{departure.toLocaleString()}</Text>
      </TouchableOpacity>
      {showPicker && (
        <DateTimePicker
          value={departure}
          mode="datetime"
          display={Platform.OS === 'ios' ? 'inline' : 'default'}
          onChange={(_, selected) => {
            setShowPicker(Platform.OS === 'ios');
            if (selected) setDeparture(selected);
          }}
        />
      )}

      <TouchableOpacity
        style={[styles.submitButton, submitting && styles.submitButtonDisabled]}
        onPress={handleSubmit}
        disabled={submitting}
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.submitButtonText}>Check Delay Risk</Text>
        )}
      </TouchableOpacity>

      {error && <Text style={styles.errorText}>{error}</Text>}

      {result && (
        <View style={styles.resultCard}>
          <View style={[styles.riskBadge, { backgroundColor: RISK_COLORS[result.risk_level] }]}>
            <Text style={styles.riskBadgeText}>{result.risk_level.toUpperCase()} RISK</Text>
          </View>
          <Text style={styles.probability}>
            {Math.round(result.delay_probability * 100)}% chance of delay ≥15 min
          </Text>
          {result.caveats.map((c) => (
            <Text key={c} style={styles.caveat}>
              ⚠ {c}
            </Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, paddingTop: 60, gap: 4 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 20 },
  title: { fontSize: 28, fontWeight: '700' },
  subtitle: { fontSize: 15, color: '#666', marginBottom: 16 },
  label: { fontSize: 13, fontWeight: '600', color: '#444', marginTop: 12 },
  pickerWrapper: { borderWidth: 1, borderColor: '#ddd', borderRadius: 8, overflow: 'hidden' },
  dateButton: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginTop: 4,
  },
  submitButton: {
    backgroundColor: '#1565c0',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    marginTop: 24,
  },
  submitButtonDisabled: { opacity: 0.6 },
  submitButtonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  errorText: { color: '#c62828', marginTop: 12 },
  hint: { color: '#666' },
  resultCard: {
    marginTop: 24,
    padding: 16,
    borderRadius: 12,
    backgroundColor: '#f5f5f5',
    gap: 8,
  },
  riskBadge: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  riskBadgeText: { color: '#fff', fontWeight: '700', fontSize: 12 },
  probability: { fontSize: 18, fontWeight: '600' },
  caveat: { fontSize: 13, color: '#7a5c00' },
});
