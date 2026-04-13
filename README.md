# TCC - Modelo analítico multicritério (AHP–TOPSIS) para suporte à decisão na seleção de hardware IoT para rastreabilidade de máquinas on farm

## Visão geral

Este projeto tem como objetivo desenvolver um pipeline de dados e um modelo analítico multicritério para apoiar a seleção de hardware IoT aplicado à rastreabilidade de máquinas on farm de tratamento de sementes.

As alternativas avaliadas são:

- FMC920 (Teltonika)
- Zefiro 4G
- ABS CEL IO

A comparação considera critérios técnicos e operacionais extraídos a partir de dados brutos de telemetria, posteriormente processados em camadas de parsing, normalização e consolidação analítica.

## Problema de pesquisa

Os três hardwares avaliados apresentam diferenças relevantes de protocolo, estrutura de payload, origem de ingestão, estabilidade de comunicação e qualidade dos dados. Essas diferenças dificultam a comparação direta entre as alternativas e exigem a construção de um pipeline de integração que permita transformar dados heterogêneos em métricas comparáveis.

## Objetivo

Construir um modelo de apoio à decisão baseado em AHP–TOPSIS para comparar três soluções de telemetria IoT, considerando:

- qualidade dos dados
- conectividade
- latência
- precisão/validade do GPS
- cobertura operacional
- robustez do pipeline de ingestão

## Arquitetura do pipeline

```text
Raw (S3) → Parsing → Normalização → Métricas → AHP → TOPSIS