import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Chip,
  List,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { StackNavigationProp } from '@react-navigation/stack';

import { ps05API, HealthResponse, InfoResponse, PS05Response } from '../utils/api';

type RootStackParamList = {
  Home: undefined;
  Document: undefined;
  Results: { result: PS05Response; imageUri: string };
  History: undefined;
  Settings: undefined;
};

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Home'>;

interface HomeScreenProps {
  navigation: HomeScreenNavigationProp;
}

export default function HomeScreen({ navigation }: HomeScreenProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      setLoading(true);
      setError(null);

      const [healthData, infoData] = await Promise.all([
        ps05API.healthCheck(),
        ps05API.getInfo(),
      ]);

      setHealth(healthData);
      setInfo(infoData);
    } catch (err) {
      setError('Failed to connect to server. Make sure the backend is running.');
      console.error('Failed to load system status:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleProcessDocument = () => {
    navigation.navigate('Document');
  };

  const handleViewHistory = () => {
    navigation.navigate('History');
  };

  const handleOpenSettings = () => {
    navigation.navigate('Settings');
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return '#4CAF50';
      case 'unhealthy':
        return '#F44336';
      default:
        return '#FF9800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'healthy':
        return 'check-circle';
      case 'unhealthy':
        return 'error';
      default:
        return 'warning';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Header */}
        <Card style={styles.headerCard}>
          <Card.Content>
            <Text style={styles.title}>PS-05 Document AI</Text>
            <Text style={styles.subtitle}>Intelligent Multilingual Document Understanding</Text>
          </Card.Content>
        </Card>

        {/* System Status */}
        <Card style={styles.card}>
          <Card.Title title="System Status" />
          <Card.Content>
            {loading ? (
              <Text style={styles.loadingText}>Loading system status...</Text>
            ) : error ? (
              <View style={styles.errorContainer}>
                <MaterialIcons name="error" size={24} color="#F44336" />
                <Text style={styles.errorText}>{error}</Text>
                <Button mode="outlined" onPress={loadSystemStatus} style={styles.retryButton}>
                  Retry
                </Button>
              </View>
            ) : (
              <View>
                <List.Item
                  title="Backend Status"
                  description={health?.status || 'Unknown'}
                  left={(props) => (
                    <MaterialIcons
                      {...props}
                      name={getStatusIcon(health?.status || '')}
                      size={24}
                      color={getStatusColor(health?.status || '')}
                    />
                  )}
                  right={() => (
                    <Chip
                      mode="outlined"
                      style={{ backgroundColor: getStatusColor(health?.status || '') + '20' }}
                    >
                      {health?.status || 'Unknown'}
                    </Chip>
                  )}
                />
                {info && (
                  <>
                    <List.Item
                      title="Version"
                      description={info.version}
                      left={(props) => <MaterialIcons {...props} name="info" size={24} />}
                    />
                    <List.Item
                      title="Supported Languages"
                      description={info.supported_languages?.join(', ') || 'None'}
                      left={(props) => <MaterialIcons {...props} name="language" size={24} />}
                    />
                    <List.Item
                      title="Processing Stages"
                      description={`${info.supported_stages?.join(', ') || 'None'}`}
                      left={(props) => <MaterialIcons {...props} name="layers" size={24} />}
                    />
                  </>
                )}
              </View>
            )}
          </Card.Content>
        </Card>

        {/* Quick Actions */}
        <Card style={styles.card}>
          <Card.Title title="Quick Actions" />
          <Card.Content>
            <View style={styles.actionButtons}>
              <Button
                mode="contained"
                icon="camera"
                onPress={handleProcessDocument}
                style={styles.actionButton}
                disabled={loading || !!error}
              >
                Process Document
              </Button>
              <Button
                mode="outlined"
                icon="history"
                onPress={handleViewHistory}
                style={styles.actionButton}
              >
                View History
              </Button>
              <Button
                mode="outlined"
                icon="settings"
                onPress={handleOpenSettings}
                style={styles.actionButton}
              >
                Settings
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Features */}
        <Card style={styles.card}>
          <Card.Title title="Features" />
          <Card.Content>
            <List.Item
              title="Layout Detection"
              description="Detect text, titles, lists, tables, and figures"
              left={(props) => <MaterialIcons {...props} name="view-column" size={24} />}
            />
            <List.Item
              title="Multilingual OCR"
              description="Support for English, Hindi, Urdu, Arabic, Nepali, Persian"
              left={(props) => <MaterialIcons {...props} name="translate" size={24} />}
            />
            <List.Item
              title="Language Identification"
              description="Automatically detect text language"
              left={(props) => <MaterialIcons {...props} name="language" size={24} />}
            />
            <List.Item
              title="Natural Language Generation"
              description="Generate summaries for tables, charts, and figures"
              left={(props) => <MaterialIcons {...props} name="text-fields" size={24} />}
            />
          </Card.Content>
        </Card>

        {/* Processing Stages */}
        <Card style={styles.card}>
          <Card.Title title="Processing Stages" />
          <Card.Content>
            <List.Item
              title="Stage 1: Layout Detection"
              description="Detect document layout elements with bounding boxes"
              left={(props) => <MaterialIcons {...props} name="view-column" size={24} />}
            />
            <List.Item
              title="Stage 2: OCR & Language ID"
              description="Extract text with language identification"
              left={(props) => <MaterialIcons {...props} name="text-fields" size={24} />}
            />
            <List.Item
              title="Stage 3: Full Analysis"
              description="Complete analysis with natural language descriptions"
              left={(props) => <MaterialIcons {...props} name="analytics" size={24} />}
            />
          </Card.Content>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollView: {
    flex: 1,
  },
  headerCard: {
    margin: 16,
    elevation: 2,
    backgroundColor: '#2196F3',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 14,
    color: 'white',
    textAlign: 'center',
    marginTop: 4,
  },
  card: {
    margin: 16,
    elevation: 2,
  },
  loadingText: {
    textAlign: 'center',
    color: '#666',
    fontStyle: 'italic',
  },
  errorContainer: {
    alignItems: 'center',
    padding: 16,
  },
  errorText: {
    color: '#F44336',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  retryButton: {
    marginTop: 8,
  },
  actionButtons: {
    flexDirection: 'column',
    gap: 12,
  },
  actionButton: {
    marginBottom: 8,
  },
});
