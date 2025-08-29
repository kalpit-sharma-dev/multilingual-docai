import React, { useState } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Alert,
  Image,
  Dimensions,
} from 'react-native';
import {
  Text,
  Card,
  Button,
  Chip,
  ProgressBar,
  ActivityIndicator,
  Divider,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { StackNavigationProp } from '@react-navigation/stack';

import ps05API, { PS05Response } from '../utils/api';

const { width } = Dimensions.get('window');

type RootStackParamList = {
  Home: undefined;
  Document: undefined;
  Results: { result: PS05Response; imageUri: string };
  History: undefined;
  Settings: undefined;
};

type DocumentScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Document'>;

interface DocumentScreenProps {
  navigation: DocumentScreenNavigationProp;
}

export default function DocumentScreen({ navigation }: DocumentScreenProps) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState(3);

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant camera roll permissions to select images.'
      );
      return false;
    }
    return true;
  };

  const pickImage = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    try {
      const imageResult = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!imageResult.canceled && imageResult.assets[0]) {
        setSelectedImage(imageResult.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick image');
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please grant camera permissions to take photos.'
      );
      return;
    }

    try {
      const photoResult = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!photoResult.canceled && photoResult.assets[0]) {
        setSelectedImage(photoResult.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to take photo');
    }
  };

  const pickDocument = async () => {
    try {
      const docResult = await DocumentPicker.getDocumentAsync({
        type: 'image/*',
        copyToCacheDirectory: true,
      });

      if (!docResult.canceled && docResult.assets[0]) {
        setSelectedImage(docResult.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick document');
    }
  };

  const processDocument = async () => {
    if (!selectedImage) {
      Alert.alert('Error', 'Please select an image first');
      return;
    }

    setProcessing(true);
    setProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 0.9) {
            clearInterval(progressInterval);
            return 0.9;
          }
          return prev + 0.1;
        });
      }, 200);

      const apiResponse = await ps05API.processDocument(selectedImage, stage);
      
      clearInterval(progressInterval);
      setProgress(1);

      // Navigate to results after a short delay
      setTimeout(() => {
        navigation.navigate('Results', { result: apiResponse, imageUri: selectedImage });
      }, 500);

    } catch (error) {
      Alert.alert('Processing Error', error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setProcessing(false);
      setProgress(0);
    }
  };

  const getStageDescription = (stage: number) => {
    switch (stage) {
      case 1:
        return 'Layout Detection Only';
      case 2:
        return 'Layout + OCR + Language ID';
      case 3:
        return 'Full Analysis (Layout + OCR + NL Generation)';
      default:
        return 'Unknown Stage';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        {/* Image Selection */}
        <Card style={styles.card}>
          <Card.Title title="Select Document" />
          <Card.Content>
            {selectedImage ? (
              <View style={styles.imageContainer}>
                <Image source={{ uri: selectedImage }} style={styles.image} />
                <Button
                  mode="outlined"
                  onPress={() => setSelectedImage(null)}
                  style={styles.removeButton}
                >
                  Remove Image
                </Button>
              </View>
            ) : (
              <View style={styles.selectionButtons}>
                <Button
                  mode="outlined"
                  icon="camera"
                  onPress={takePhoto}
                  style={styles.selectionButton}
                >
                  Take Photo
                </Button>
                <Button
                  mode="outlined"
                  icon="image"
                  onPress={pickImage}
                  style={styles.selectionButton}
                >
                  Choose from Gallery
                </Button>
                <Button
                  mode="outlined"
                  icon="file-document"
                  onPress={pickDocument}
                  style={styles.selectionButton}
                >
                  Pick Document
                </Button>
              </View>
            )}
          </Card.Content>
        </Card>

        {/* Processing Options */}
        <Card style={styles.card}>
          <Card.Title title="Processing Options" />
          <Card.Content>
            <Text style={styles.optionLabel}>Processing Stage:</Text>
            <View style={styles.stageButtons}>
              {[1, 2, 3].map((stageNum) => (
                <Chip
                  key={stageNum}
                  selected={stage === stageNum}
                  onPress={() => setStage(stageNum)}
                  style={styles.stageChip}
                  mode={stage === stageNum ? 'flat' : 'outlined'}
                >
                  Stage {stageNum}
                </Chip>
              ))}
            </View>
            <Text style={styles.stageDescription}>
              {getStageDescription(stage)}
            </Text>
          </Card.Content>
        </Card>

        {/* Processing Status */}
        {processing && (
          <Card style={styles.card}>
            <Card.Title title="Processing Document" />
            <Card.Content>
              <View style={styles.progressContainer}>
                <ProgressBar progress={progress} color="#2196F3" />
                <Text style={styles.progressText}>
                  {Math.round(progress * 100)}% Complete
                </Text>
              </View>
              <View style={styles.processingInfo}>
                <ActivityIndicator size="small" color="#2196F3" />
                <Text style={styles.processingText}>
                  Analyzing document... Please wait.
                </Text>
              </View>
            </Card.Content>
          </Card>
        )}

        {/* Process Button */}
        <Card style={styles.card}>
          <Card.Content>
            <Button
              mode="contained"
              icon="play"
              onPress={processDocument}
              disabled={!selectedImage || processing}
              loading={processing}
              style={styles.processButton}
            >
              {processing ? 'Processing...' : 'Process Document'}
            </Button>
          </Card.Content>
        </Card>

        {/* Quick Info */}
        <Card style={styles.card}>
          <Card.Title title="Supported Features" />
          <Card.Content>
            <View style={styles.featureItem}>
              <MaterialIcons name="language" size={20} color="#2196F3" />
              <Text style={styles.featureText}>
                Multilingual OCR (English, Hindi, Urdu, Arabic, Nepali, Persian)
              </Text>
            </View>
            <Divider style={styles.divider} />
            <View style={styles.featureItem}>
              <MaterialIcons name="view-column" size={20} color="#2196F3" />
              <Text style={styles.featureText}>
                Layout detection (Text, Title, List, Table, Figure)
              </Text>
            </View>
            <Divider style={styles.divider} />
            <View style={styles.featureItem}>
              <MaterialIcons name="description" size={20} color="#2196F3" />
              <Text style={styles.featureText}>
                Natural language summaries for tables, charts, and maps
              </Text>
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
  imageContainer: {
    alignItems: 'center',
  },
  image: {
    width: width - 64,
    height: (width - 64) * 0.75,
    borderRadius: 8,
    marginBottom: 16,
  },
  removeButton: {
    marginTop: 8,
  },
  selectionButtons: {
    gap: 12,
  },
  selectionButton: {
    marginBottom: 8,
  },
  optionLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#212121',
    marginBottom: 12,
  },
  stageButtons: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  stageChip: {
    marginRight: 8,
  },
  stageDescription: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
  },
  progressContainer: {
    marginBottom: 16,
  },
  progressText: {
    textAlign: 'center',
    marginTop: 8,
    fontSize: 14,
    color: '#666',
  },
  processingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  processingText: {
    marginLeft: 8,
    fontSize: 14,
    color: '#666',
  },
  processButton: {
    marginTop: 8,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
  },
  featureText: {
    fontSize: 14,
    color: '#212121',
    marginLeft: 12,
    flex: 1,
  },
  divider: {
    marginVertical: 8,
  },
}); 