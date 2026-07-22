import { StatusBar } from 'expo-status-bar';
import { SafeAreaView, StyleSheet } from 'react-native';
import PredictScreen from './src/screens/PredictScreen';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <PredictScreen />
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
