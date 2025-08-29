import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Card, Text, Chip, Button } from 'react-native-paper';
import { MaterialIcons } from '@expo/vector-icons';

interface DocCardProps {
  title: string;
  subtitle?: string;
  status: 'completed' | 'failed' | 'processing';
  stage: number;
  processingTime?: number;
  elements?: number;
  textLines?: number;
  tables?: number;
  charts?: number;
  onPress?: () => void;
  onShare?: () => void;
  onDelete?: () => void;
}

export default function DocCard({
  title,
  subtitle,
  status,
  stage,
  processingTime,
  elements,
  textLines,
  tables,
  charts,
  onPress,
  onShare,
  onDelete,
}: DocCardProps) {
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

  return (
    <Card style={styles.card} onPress={onPress}>
      <Card.Content>
        <View style={styles.header}>
          <View style={styles.titleContainer}>
            <Text style={styles.title}>{title}</Text>
            {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
          </View>
          <View style={styles.statusContainer}>
            <MaterialIcons
              name={getStatusIcon(status)}
              size={24}
              color={getStatusColor(status)}
            />
            <Chip
              mode="outlined"
              textStyle={{ color: getStatusColor(status) }}
              style={[styles.statusChip, { borderColor: getStatusColor(status) }]}
            >
              {status}
            </Chip>
          </View>
        </View>

        <View style={styles.details}>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Stage:</Text>
            <Chip mode="outlined" style={styles.stageChip}>
              {getStageDescription(stage)}
            </Chip>
          </View>
          
          {processingTime && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Time:</Text>
              <Text style={styles.detailValue}>{processingTime}s</Text>
            </View>
          )}

          {elements !== undefined && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Elements:</Text>
              <Text style={styles.detailValue}>{elements}</Text>
            </View>
          )}

          {textLines !== undefined && textLines > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Text Lines:</Text>
              <Text style={styles.detailValue}>{textLines}</Text>
            </View>
          )}

          {tables !== undefined && tables > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Tables:</Text>
              <Text style={styles.detailValue}>{tables}</Text>
            </View>
          )}

          {charts !== undefined && charts > 0 && (
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Charts:</Text>
              <Text style={styles.detailValue}>{charts}</Text>
            </View>
          )}
        </View>

        {(onShare || onDelete) && (
          <View style={styles.actions}>
            {onShare && (
              <Button
                mode="outlined"
                icon="share"
                onPress={onShare}
                style={styles.actionButton}
              >
                Share
              </Button>
            )}
            {onDelete && (
              <Button
                mode="outlined"
                icon="delete"
                onPress={onDelete}
                style={styles.actionButton}
                textColor="#F44336"
              >
                Delete
              </Button>
            )}
          </View>
        )}
      </Card.Content>
    </Card>
  );
}

const styles = StyleSheet.create({
  card: {
    margin: 8,
    elevation: 2,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  titleContainer: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#212121',
  },
  subtitle: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  statusContainer: {
    alignItems: 'center',
  },
  statusChip: {
    marginTop: 4,
  },
  details: {
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
  actions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 4,
  },
});
