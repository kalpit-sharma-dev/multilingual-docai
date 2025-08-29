import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  ScrollView,
  Alert,
  Switch,
} from 'react-native';
import {
  Card,
  Button,
  List,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialIcons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface Settings {
  serverUrl: string;
  defaultStage: number;
  autoSave: boolean;
  highQuality: boolean;
  darkMode: boolean;
  notifications: boolean;
}

export default function SettingsScreen() {
  const [settings, setSettings] = useState<Settings>({
    serverUrl: 'http://localhost:8000',
    defaultStage: 3,
    autoSave: true,
    highQuality: false,
    darkMode: false,
    notifications: true,
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const savedSettings = await AsyncStorage.getItem('settings');
      if (savedSettings) {
        setSettings(JSON.parse(savedSettings));
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };

  const saveSettings = async () => {
    try {
      await AsyncStorage.setItem('settings', JSON.stringify(settings));
      Alert.alert('Success', 'Settings saved successfully');
    } catch (error) {
      Alert.alert('Error', 'Failed to save settings');
    }
  };

  const resetSettings = () => {
    Alert.alert(
      'Reset Settings',
      'Are you sure you want to reset all settings to default?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            setSettings({
              serverUrl: 'http://localhost:8000',
              defaultStage: 3,
              autoSave: true,
              highQuality: false,
              darkMode: false,
              notifications: true,
            });
          },
        },
      ]
    );
  };

  const clearCache = () => {
    Alert.alert(
      'Clear Cache',
      'Are you sure you want to clear all cached data?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              await AsyncStorage.clear();
              Alert.alert('Success', 'Cache cleared successfully');
            } catch (error) {
              Alert.alert('Error', 'Failed to clear cache');
            }
          },
        },
      ]
    );
  };

  const exportSettings = () => {
    Alert.alert('Export Settings', 'Settings export feature coming soon!');
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        <Card style={styles.card}>
          <Card.Title title="Server Configuration" />
          <Card.Content>
            <List.Item
              title="Server URL"
              description={settings.serverUrl}
              left={(props) => <MaterialIcons {...props} name="save" size={24} />}
              onPress={() => {
                // In a real app, this would open a text input dialog
                Alert.alert('Server URL', 'Enter server URL:', [
                  { text: 'Cancel', style: 'cancel' },
                  {
                    text: 'Save',
                    onPress: (text) => {
                      if (text) {
                        setSettings({ ...settings, serverUrl: text });
                      }
                    },
                  },
                ]);
              }}
            />
            <List.Item
              title="High Quality Processing"
              description="Use higher resolution processing (slower but more accurate)"
              left={(props) => <MaterialIcons {...props} name="high-quality" size={24} />}
              right={() => (
                <Switch
                  value={settings.highQuality}
                  onValueChange={(value) =>
                    setSettings({ ...settings, highQuality: value })
                  }
                />
              )}
            />
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="Processing Options" />
          <Card.Content>
            <List.Item
              title="Default Processing Stage"
              description={`Stage ${settings.defaultStage}`}
              left={(props) => <MaterialIcons {...props} name="settings" size={24} />}
              onPress={() => {
                Alert.alert(
                  'Default Stage',
                  'Select default processing stage:',
                  [
                    { text: 'Stage 1 (Layout)', onPress: () => setSettings({ ...settings, defaultStage: 1 }) },
                    { text: 'Stage 2 (OCR)', onPress: () => setSettings({ ...settings, defaultStage: 2 }) },
                    { text: 'Stage 3 (Full)', onPress: () => setSettings({ ...settings, defaultStage: 3 }) },
                    { text: 'Cancel', style: 'cancel' },
                  ]
                );
              }}
            />
            <List.Item
              title="Auto Save Results"
              description="Automatically save processing results"
              left={(props) => <MaterialIcons {...props} name="auto-fix-high" size={24} />}
              right={() => (
                <Switch
                  value={settings.autoSave}
                  onValueChange={(value) =>
                    setSettings({ ...settings, autoSave: value })
                  }
                />
              )}
            />
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="Appearance" />
          <Card.Content>
            <List.Item
              title="Dark Mode"
              description="Use dark theme"
              left={(props) => <MaterialIcons {...props} name="dark-mode" size={24} />}
              right={() => (
                <Switch
                  value={settings.darkMode}
                  onValueChange={(value) =>
                    setSettings({ ...settings, darkMode: value })
                  }
                />
              )}
            />
            <List.Item
              title="Notifications"
              description="Show processing notifications"
              left={(props) => <MaterialIcons {...props} name="notifications" size={24} />}
              right={() => (
                <Switch
                  value={settings.notifications}
                  onValueChange={(value) =>
                    setSettings({ ...settings, notifications: value })
                  }
                />
              )}
            />
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="Data Management" />
          <Card.Content>
            <List.Item
              title="Clear Cache"
              description="Clear all cached data and settings"
              left={(props) => <MaterialIcons {...props} name="clear" size={24} />}
              onPress={clearCache}
            />
            <List.Item
              title="Export Settings"
              description="Export current settings to file"
              left={(props) => <MaterialIcons {...props} name="file-download" size={24} />}
              onPress={exportSettings}
            />
            <List.Item
              title="Reset to Default"
              description="Reset all settings to default values"
              left={(props) => <MaterialIcons {...props} name="restore" size={24} />}
              onPress={resetSettings}
            />
          </Card.Content>
        </Card>

        <Card style={styles.card}>
          <Card.Title title="About" />
          <Card.Content>
            <List.Item
              title="App Version"
              description="1.0.0"
              left={(props) => <MaterialIcons {...props} name="info" size={24} />}
            />
            <List.Item
              title="Build Number"
              description="2024.01.15"
              left={(props) => <MaterialIcons {...props} name="build" size={24} />}
            />
            <List.Item
              title="License"
              description="MIT License"
              left={(props) => <MaterialIcons {...props} name="gavel" size={24} />}
            />
          </Card.Content>
        </Card>

        <View style={styles.buttonContainer}>
          <Button
            mode="contained"
            onPress={saveSettings}
            style={styles.saveButton}
          >
            Save Settings
          </Button>
        </View>
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
  buttonContainer: {
    padding: 16,
  },
  saveButton: {
    marginTop: 8,
  },
}); 