# Dicionário de dados da base normalizada

## Objetivo
A base normalizada consolida os dados provenientes dos hardwares FMC920, Zefiro 4G e ABS CEL IO em um schema canônico único, permitindo análise comparativa e cálculo de métricas homogêneas.

## Grão da base
Cada linha representa um evento de telemetria individual, já parseado e semanticamente normalizado, associado a uma sessão operacional.

## Campos

### Identificação
- `normalized_id`: identificador único do registro normalizado
- `hardware`: hardware de origem (`FMC920`, `ZEFIRO`, `ABS`)
- `device_id`: identificador do dispositivo
- `hardware_session_id`: identificador técnico da sessão, quando aplicável
- `session_id`: identificador da sessão operacional
- `source_system`: origem de ingestão
- `payload_format`: formato bruto (`json`, `ascii`, `hex`)

### Contexto operacional
- `state_client`: estado do cliente
- `city`: cidade do cliente
- `execution_place`: local da operação
- `seed_type`: tipo de semente
- `weight_kg`: peso associado à operação

### Tempo
- `event_time`: horário do evento no dispositivo
- `ingestion_time`: horário de recepção no gateway
- `event_date`: data derivada
- `event_hour`: hora derivada
- `delay_ms`: latência estimada
- `is_late`: indicador de atraso
- `is_buffered`: indicador de buffer/reenvio

### Posicionamento e conectividade
- `latitude`: latitude normalizada
- `longitude`: longitude normalizada
- `latitude_ref`: latitude de referência
- `longitude_ref`: longitude de referência
- `gps_valid`: indicador de validade do GPS
- `gps_status`: status semântico do GPS
- `speed_kmh`: velocidade em km/h
- `rssi_dbm`: intensidade do sinal
- `signal_quality`: categoria da qualidade de sinal
- `network_condition`: condição de rede

### Estado da máquina
- `machine_on`: máquina ligada
- `machine_operating`: máquina operando
- `machine_status`: estado consolidado da máquina
- `scenario_state`: estado contextual da simulação/operação

### Qualidade e rastreabilidade
- `parse_ok`: status do parsing
- `parse_error`: descrição de erro de parsing
- `data_quality_ok`: status da validação mínima
- `data_quality_flags`: flags de inconsistência
- `raw_record_id`: referência ao registro bruto
- `raw_payload`: payload bruto preservado
- `observacoes`: observações da camada de ingestão