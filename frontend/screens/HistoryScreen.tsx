import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Alert,
  RefreshControl,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Chip,
  Searchbar,
  FAB,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import { StackNavigationProp } from '@react-navigation/stack';
import { PS05Response } from '../utils/api';

// Mock data for demonstration
const mockHistory: HistoryItem[] = [
  {
    id: '1',
    filename: 'document1.pdf',
    timestamp: new Date('2024-01-15T10:30:00'),
    stage: 3,
    status: 'completed',
    processingTime: 2.5,
    elements: 15,
    textLines: 45,
    tables: 2,
    charts: 1,
  },
  {
    id: '2',
    filename: 'report2.jpg',
    timestamp: new Date('2024-01-14T15:45:00'),
    stage: 2,
    status: 'completed',
    processingTime: 1.8,
    elements: 8,
    textLines: 23,
    tables: 1,
    charts: 0,
  },
  {
    id: '3',
    filename: 'chart3.png',
    timestamp: new Date('2024-01-13T09:20:00'),
    stage: 1,
    status: 'failed',
    processingTime: 0.5,
    elements: 0,
    textLines: 0,
    tables: 0,
    charts: 0,
  },
];

interface HistoryItem {
  id: string;
  filename: string;
  timestamp: Date;
  stage: number;
  status: 'completed' | 'failed' | 'processing';
  processingTime: number;
  elements: number;
  textLines: number;
  tables: number;
  charts: number;
}

type RootStackParamList = {
  Home: undefined;
  Document: undefined;
  Results: { result: PS05Response; imageUri: string };
  History: undefined;
  Settings: undefined;
};

type HistoryScreenNavigationProp = StackNavigationProp<RootStackParamList, 'History'>;

interface HistoryScreenProps {
  navigation: HistoryScreenNavigationProp;
}

export default function HistoryScreen({ navigation }: HistoryScreenProps) {
  const [history, setHistory] = useState<HistoryItem[]>(mockHistory);
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>(mockHistory);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState('all');

  const onRefresh = () => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setHistory(mockHistory);
      setFilteredHistory(mockHistory);
      setRefreshing(false);
    }, 1000);
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    filterHistory(query, selectedFilter);
  };

  const handleFilter = (filter: string) => {
    setSelectedFilter(filter);
    filterHistory(searchQuery, filter);
  };

  const filterHistory = (query: string, filter: string) => {
    let filtered = history;

    // Apply search filter
    if (query) {
      filtered = filtered.filter(item =>
        item.filename.toLowerCase().includes(query.toLowerCase())
      );
    }

    // Apply status filter
    if (filter !== 'all') {
      filtered = filtered.filter(item => item.status === filter);
    }

    setFilteredHistory(filtered);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#4CAF50';
      case 'failed':
        return '#F44336';
      case 'processing':
        return '#FF9800';
      default:
        return '#666';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return 'check-circle';
      case 'failed':
        return 'error';
      case 'processing':
        return 'hourglass-empty';
      default:
        return 'help';
    }
  };

  const getStageDescription = (stage: number) => {
    switch (stage) {
      case 1:
        return 'Layout Only';
      case 2:
        return 'Layout + OCR';
      case 3:
        return 'Full Analysis';
      default:
        return 'Unknown';
    }
  };

  const deleteItem = (id: string) => {
    Alert.alert(
      'Delete Item',
      'Are you sure you want to delete this item from history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            const updatedHistory = history.filter(item => item.id !== id);
            setHistory(updatedHistory);
            filterHistory(searchQuery, selectedFilter);
          },
        },
      ]
    );
  };

  const clearHistory = () => {
    Alert.alert(
      'Clear History',
      'Are you sure you want to clear all history?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: () => {
            setHistory([]);
            setFilteredHistory([]);
          },
        },
      ]
    );
  };

  const exportHistory = () => {
    const historyData = JSON.stringify(history, null, 2);
    Alert.alert(
      'Export History',
      `History exported:\n${historyData}`
    );
  };

  const renderHistoryItem = (item: HistoryItem) => (
    <Card key={item.id} style={styles.historyCard}>
      <Card.Content>
        <View style={styles.itemHeader}>
          <View style={styles.itemInfo}>
            <Text style={styles.filename}>{item.filename}</Text>
            <Text style={styles.timestamp}>
              {item.timestamp.toLocaleDateString()} {item.timestamp.toLocaleTimeString()}
            </Text>
          </View>
          <View style={styles.itemStatus}>
            <MaterialIcons
              name={getStatusIcon(item.status)}
              size={24}
              color={getStatusColor(item.status)}
            />
            <Chip
              mode="outlined"
              textStyle={{ color: getStatusColor(item.status) }}
              style={[
                styles.statusChip,
                { borderColor: getStatusColor(item.status) },
              ]}
            >
              {item.status}
            </Chip>
          </View>
        </View>

        <View style={styles.itemDetails}>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Stage:</Text>
            <Chip mode="outlined" style={styles.stageChip}>
              {getStageDescription(item.stage)}
            </Chip>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Processing Time:</Text>
            <Text style={styles.detailValue}>{item.processingTime}s</Text>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Elements:</Text>
            <Text style={styles.detailValue}>{item.elements}</Text>
          </View>
          {item.textLines > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Text Lines:</Text>
              <Text style={styles.detailValue}>{item.textLines}</Text>
            </View>
          )}
          {item.tables > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Tables:</Text>
              <Text style={styles.detailValue}>{item.tables}</Text>
            </View>
          )}
          {item.charts > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Charts:</Text>
              <Text style={styles.detailValue}>{item.charts}</Text>
            </View>
          )}
        </View>

        <View style={styles.itemActions}>
          <Button
            mode="outlined"
            icon="eye"
            onPress={() => Alert.alert('View Results', 'View results feature coming soon!')}
            style={styles.actionButton}
          >
            View Results
          </Button>
          <Button
            mode="outlined"
            icon="share"
            onPress={() => Alert.alert('Share', 'Sharing results...')}
            style={styles.actionButton}
          >
            Share
          </Button>
          <Button
            mode="outlined"
            icon="delete"
            onPress={() => deleteItem(item.id)}
            style={styles.actionButton}
            textColor="#F44336"
          >
            Delete
          </Button>
        </View>
      </Card.Content>
    </Card>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Search and Filters */}
        <Card style={styles.card}>
          <Card.Content>
            <Searchbar
              placeholder="Search documents..."
              onChangeText={handleSearch}
              value={searchQuery}
              style={styles.searchbar}
            />
            <View style={styles.filterContainer}>
              {[
                { key: 'all', label: 'All' },
                { key: 'completed', label: 'Completed' },
                { key: 'failed', label: 'Failed' },
                { key: 'processing', label: 'Processing' },
              ].map((filter) => (
                <Chip
                  key={filter.key}
                  selected={selectedFilter === filter.key}
                  onPress={() => handleFilter(filter.key)}
                  style={styles.filterChip}
                  mode={selectedFilter === filter.key ? 'flat' : 'outlined'}
                >
                  {filter.label}
                </Chip>
              ))}
            </View>
          </Card.Content>
        </Card>

        {/* Statistics */}
        <Card style={styles.card}>
          <Card.Title title="Statistics" />
          <Card.Content>
            <View style={styles.statsContainer}>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>{history.length}</Text>
                <Text style={styles.statLabel}>Total Documents</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>
                  {history.filter(item => item.status === 'completed').length}
                </Text>
                <Text style={styles.statLabel}>Completed</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>
                  {history.filter(item => item.status === 'failed').length}
                </Text>
                <Text style={styles.statLabel}>Failed</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statNumber}>
                  {history.reduce((sum, item) => sum + item.processingTime, 0).toFixed(1)}s
                </Text>
                <Text style={styles.statLabel}>Total Time</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* History List */}
        {filteredHistory.length === 0 ? (
          <Card style={styles.card}>
            <Card.Content>
              <View style={styles.emptyContainer}>
                <MaterialIcons name="history" size={48} color="#666" />
                <Text style={styles.emptyText}>No documents in history</Text>
                <Text style={styles.emptySubtext}>
                  Process some documents to see them here
                </Text>
              </View>
            </Card.Content>
          </Card>
        ) : (
          filteredHistory.map(renderHistoryItem)
        )}

        {/* Action Buttons */}
        {history.length > 0 && (
          <Card style={styles.card}>
            <Card.Content>
              <View style={styles.actionButtons}>
                <Button
                  mode="outlined"
                  icon="file-download"
                  onPress={exportHistory}
                  style={styles.actionButton}
                >
                  Export History
                </Button>
                <Button
                  mode="outlined"
                  icon="delete-sweep"
                  onPress={clearHistory}
                  style={styles.actionButton}
                  textColor="#F44336"
                >
                  Clear All
                </Button>
              </View>
            </Card.Content>
          </Card>
        )}
      </ScrollView>

      <FAB
        style={styles.fab}
        icon="plus"
        onPress={() => navigation.navigate('Document')}
        label="Process New Document"
      />
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
  card: {
    margin: 16,
    elevation: 2,
  },
  searchbar: {
    marginBottom: 16,
  },
  filterContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  filterChip: {
    marginRight: 8,
    marginBottom: 4,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2196F3',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
  },
  historyCard: {
    margin: 16,
    elevation: 2,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  itemInfo: {
    flex: 1,
  },
  filename: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#212121',
  },
  timestamp: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  itemStatus: {
    alignItems: 'center',
  },
  statusChip: {
    marginTop: 4,
  },
  itemDetails: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    color: '#212121',
    fontWeight: 'bold',
  },
  stageChip: {
    alignSelf: 'flex-start',
  },
  itemActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 4,
  },
  emptyContainer: {
    alignItems: 'center',
    padding: 32,
  },
  emptyText: {
    fontSize: 18,
    color: '#666',
    marginTop: 16,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  fab: {
    position: 'absolute',
    margin: 16,
    right: 0,
    bottom: 0,
  },
}); 