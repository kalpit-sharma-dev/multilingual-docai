import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Image,
  Dimensions,
  Alert,
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
import * as Sharing from 'expo-sharing';
import { StackNavigationProp } from '@react-navigation/stack';
import { RouteProp } from '@react-navigation/native';

import { PS05Response } from '../utils/api';

const { width } = Dimensions.get('window');

type RootStackParamList = {
  Home: undefined;
  Document: undefined;
  Results: { result: PS05Response; imageUri: string };
  History: undefined;
  Settings: undefined;
};

type ResultsScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Results'>;
type ResultsScreenRouteProp = RouteProp<RootStackParamList, 'Results'>;

interface ResultsScreenProps {
  navigation?: ResultsScreenNavigationProp;
  route?: ResultsScreenRouteProp;
}

export default function ResultsScreen({ navigation, route }: ResultsScreenProps = {}) {
  const { result, imageUri } = route?.params || { result: null, imageUri: '' };
  const [activeTab, setActiveTab] = useState('overview');

  // Early return if no result
  if (!result) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centerContent}>
          <Text>No results available</Text>
        </View>
      </SafeAreaView>
    );
  }

  const shareResults = async () => {
    try {
      await Sharing.shareAsync(imageUri, {
        mimeType: 'image/jpeg',
        dialogTitle: 'Share Document Analysis Results',
      });
    } catch (error) {
      Alert.alert('Error', 'Failed to share results');
    }
  };

  const getElementIcon = (cls: string) => {
    switch (cls.toLowerCase()) {
      case 'text':
        return 'text-fields';
      case 'title':
        return 'title';
      case 'list':
        return 'format-list-bulleted';
      case 'table':
        return 'table-chart';
      case 'figure':
        return 'image';
      default:
        return 'help';
    }
  };

  const getElementColor = (cls: string) => {
    switch (cls.toLowerCase()) {
      case 'text':
        return '#2196F3';
      case 'title':
        return '#FF9800';
      case 'list':
        return '#4CAF50';
      case 'table':
        return '#9C27B0';
      case 'figure':
        return '#F44336';
      default:
        return '#666';
    }
  };

  const formatBbox = (bbox: [number, number, number, number]) => {
    return `[${bbox[0]}, ${bbox[1]}, ${bbox[2]}, ${bbox[3]}]`;
  };

  const renderOverview = () => (
    <View>
      <Card style={styles.card}>
        <Card.Title title="Document Information" />
        <Card.Content>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Page Size:</Text>
            <Text style={styles.infoValue}>
              {result.size.w} × {result.size.h} pixels
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Processing Time:</Text>
            <Text style={styles.infoValue}>
              {result.processing_time.toFixed(2)} seconds
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Deskew Angle:</Text>
            <Text style={styles.infoValue}>
              {result.preprocess.deskew_angle.toFixed(1)}°
            </Text>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Detection Summary" />
        <Card.Content>
          <View style={styles.summaryRow}>
            <Text style={styles.summaryLabel}>Layout Elements:</Text>
            <Text style={styles.summaryValue}>{result.elements.length}</Text>
          </View>
          {result.text_lines && (
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Text Lines:</Text>
              <Text style={styles.summaryValue}>{result.text_lines.length}</Text>
            </View>
          )}
          {result.tables && (
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Tables:</Text>
              <Text style={styles.summaryValue}>{result.tables.length}</Text>
            </View>
          )}
          {result.charts && (
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Charts:</Text>
              <Text style={styles.summaryValue}>{result.charts.length}</Text>
            </View>
          )}
          {result.figures && (
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Figures:</Text>
              <Text style={styles.summaryValue}>{result.figures.length}</Text>
            </View>
          )}
        </Card.Content>
      </Card>
    </View>
  );

  const renderLayout = () => (
    <Card style={styles.card}>
      <Card.Title title="Layout Elements" />
      <Card.Content>
        {result.elements.length === 0 ? (
          <Text style={styles.emptyText}>No layout elements detected</Text>
        ) : (
          result.elements.map((element, index) => (
            <List.Item
              key={index}
              title={element.cls}
              description={`BBox: ${formatBbox(element.bbox)} | Score: ${(element.score * 100).toFixed(1)}%`}
              left={(props) => (
                <MaterialIcons
                  {...props}
                  name={getElementIcon(element.cls)}
                  size={24}
                  color={getElementColor(element.cls)}
                />
              )}
              right={() => (
                <Chip mode="outlined" style={{ backgroundColor: getElementColor(element.cls) + '20' }}>
                  {(element.score * 100).toFixed(0)}%
                </Chip>
              )}
              style={styles.listItem}
            />
          ))
        )}
      </Card.Content>
    </Card>
  );

  const renderText = () => (
    <Card style={styles.card}>
      <Card.Title title="Text Lines" />
      <Card.Content>
        {!result.text_lines || result.text_lines.length === 0 ? (
          <Text style={styles.emptyText}>No text lines detected</Text>
        ) : (
          result.text_lines.map((line, index) => (
            <List.Item
              key={index}
              title={line.text}
              description={`Language: ${line.lang} | BBox: ${formatBbox(line.bbox)}`}
              left={(props) => (
                <MaterialIcons
                  {...props}
                  name="text-fields"
                  size={24}
                  color="#2196F3"
                />
              )}
              right={() => (
                <View style={styles.textLineInfo}>
                  <Chip mode="outlined" style={styles.langChip}>
                    {line.lang}
                  </Chip>
                  <Text style={styles.confidenceText}>
                    {(line.score * 100).toFixed(0)}%
                  </Text>
                </View>
              )}
              style={styles.listItem}
            />
          ))
        )}
      </Card.Content>
    </Card>
  );

  const renderTables = () => (
    <Card style={styles.card}>
      <Card.Title title="Tables" />
      <Card.Content>
        {!result.tables || result.tables.length === 0 ? (
          <Text style={styles.emptyText}>No tables detected</Text>
        ) : (
          result.tables.map((table, index) => (
            <List.Item
              key={index}
              title={`Table ${index + 1}`}
              description={table.summary}
              left={(props) => (
                <MaterialIcons
                  {...props}
                  name="table-chart"
                  size={24}
                  color="#9C27B0"
                />
              )}
              right={() => (
                <Chip mode="outlined" style={{ backgroundColor: '#9C27B0' + '20' }}>
                  {(table.confidence * 100).toFixed(0)}%
                </Chip>
              )}
              style={styles.listItem}
            />
          ))
        )}
      </Card.Content>
    </Card>
  );

  const renderCharts = () => (
    <Card style={styles.card}>
      <Card.Title title="Charts" />
      <Card.Content>
        {!result.charts || result.charts.length === 0 ? (
          <Text style={styles.emptyText}>No charts detected</Text>
        ) : (
          result.charts.map((chart, index) => (
            <List.Item
              key={index}
              title={`${chart.type} Chart`}
              description={chart.summary}
              left={(props) => (
                <MaterialIcons
                  {...props}
                  name="bar-chart"
                  size={24}
                  color="#FF9800"
                />
              )}
              right={() => (
                <Chip mode="outlined" style={{ backgroundColor: '#FF9800' + '20' }}>
                  {(chart.confidence * 100).toFixed(0)}%
                </Chip>
              )}
              style={styles.listItem}
            />
          ))
        )}
      </Card.Content>
    </Card>
  );

  const renderFigures = () => (
    <Card style={styles.card}>
      <Card.Title title="Figures" />
      <Card.Content>
        {!result.figures || result.figures.length === 0 ? (
          <Text style={styles.emptyText}>No figures detected</Text>
        ) : (
          result.figures.map((figure, index) => (
            <List.Item
              key={index}
              title={`Figure ${index + 1}`}
              description={figure.summary}
              left={(props) => (
                <MaterialIcons
                  {...props}
                  name="image"
                  size={24}
                  color="#F44336"
                />
              )}
              right={() => (
                <Chip mode="outlined" style={{ backgroundColor: '#F44336' + '20' }}>
                  {(figure.confidence * 100).toFixed(0)}%
                </Chip>
              )}
              style={styles.listItem}
            />
          ))
        )}
      </Card.Content>
    </Card>
  );

  const renderRawData = () => (
    <Card style={styles.card}>
      <Card.Title title="Raw JSON Data" />
      <Card.Content>
        <ScrollView style={styles.jsonContainer}>
          <Text style={styles.jsonText}>
            {JSON.stringify(result, null, 2)}
          </Text>
        </ScrollView>
      </Card.Content>
    </Card>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview();
      case 'layout':
        return renderLayout();
      case 'text':
        return renderText();
      case 'tables':
        return renderTables();
      case 'charts':
        return renderCharts();
      case 'figures':
        return renderFigures();
      case 'raw':
        return renderRawData();
      default:
        return renderOverview();
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Original Image */}
        <Card style={styles.card}>
          <Card.Title title="Original Document" />
          <Card.Content>
            <Image source={{ uri: imageUri }} style={styles.image} />
          </Card.Content>
        </Card>

        {/* Tab Navigation */}
        <Card style={styles.card}>
          <Card.Content>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              <View style={styles.tabContainer}>
                {[
                  { key: 'overview', label: 'Overview', icon: 'dashboard' },
                  { key: 'layout', label: 'Layout', icon: 'view-column' },
                  { key: 'text', label: 'Text', icon: 'text-fields' },
                  { key: 'tables', label: 'Tables', icon: 'table-chart' },
                  { key: 'charts', label: 'Charts', icon: 'bar-chart' },
                  { key: 'figures', label: 'Figures', icon: 'image' },
                  { key: 'raw', label: 'Raw Data', icon: 'code' },
                ].map((tab) => (
                  <Chip
                    key={tab.key}
                    selected={activeTab === tab.key}
                    onPress={() => setActiveTab(tab.key)}
                    style={styles.tabChip}
                    mode={activeTab === tab.key ? 'flat' : 'outlined'}
                    icon={tab.icon}
                  >
                    {tab.label}
                  </Chip>
                ))}
              </View>
            </ScrollView>
          </Card.Content>
        </Card>

        {/* Tab Content */}
        {renderTabContent()}

        {/* Action Buttons */}
        <Card style={styles.card}>
          <Card.Content>
            <View style={styles.actionButtons}>
              <Button
                mode="outlined"
                icon="share"
                onPress={shareResults}
                style={styles.actionButton}
              >
                Share Results
              </Button>
              <Button
                mode="contained"
                icon="refresh"
                onPress={() => navigation?.navigate('Document')}
                style={styles.actionButton}
              >
                Process Another
              </Button>
            </View>
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
  card: {
    margin: 16,
    elevation: 2,
  },
  image: {
    width: width - 64,
    height: (width - 64) * 0.75,
    borderRadius: 8,
  },
  tabContainer: {
    flexDirection: 'row',
    paddingVertical: 8,
  },
  tabChip: {
    marginRight: 8,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  infoLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
  },
  infoValue: {
    fontSize: 14,
    color: '#212121',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#666',
  },
  summaryValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#212121',
  },
  listItem: {
    paddingVertical: 4,
  },
  textLineInfo: {
    alignItems: 'flex-end',
  },
  langChip: {
    marginBottom: 4,
  },
  confidenceText: {
    fontSize: 12,
    color: '#666',
  },
  emptyText: {
    textAlign: 'center',
    color: '#666',
    fontStyle: 'italic',
    padding: 16,
  },
  jsonContainer: {
    maxHeight: 400,
    backgroundColor: '#F5F5F5',
    borderRadius: 4,
    padding: 8,
  },
  jsonText: {
    fontSize: 12,
    fontFamily: 'monospace',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 8,
  },
  centerContent: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
}); 