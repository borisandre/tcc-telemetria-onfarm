from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAW_ROOT = Path("data/raw_samples/raw/zefiro")
OUTPUT_DIR = Path("data/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "zefiro_parsed.parquet"
OUTPUT_CSV = OUTPUT_DIR / "zefiro_parsed.csv"


# ============================================================
# UTILITÁRIOS
# ============================================================

def safe_get(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return None

    if "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return None


def to_int01(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "sim"}:
        return 1
    if s in {"0", "false", "no", "nao", "não"}:
        return 0
    return None


def parse_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, format="ISO8601", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def classify_signal_quality(rssi_dbm: Optional[float]) -> Optional[str]:
    if rssi_dbm is None:
        return None
    if rssi_dbm >= -80:
        return "good"
    if rssi_dbm >= -90:
        return "moderate"
    if rssi_dbm >= -100:
        return "poor"
    return "critical"


def derive_machine_status(machine_on: Optional[int], machine_operating: Optional[int]) -> Optional[str]:
    if machine_on is None or machine_operating is None:
        return None
    if machine_on == 0:
        return "off"
    if machine_on == 1 and machine_operating == 0:
        return "idle"
    if machine_on == 1 and machine_operating == 1:
        return "operating"
    return None


def make_normalized_id() -> str:
    return str(uuid.uuid4())


def build_quality_flags(
    event_time: Optional[pd.Timestamp],
    device_id: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    machine_on: Optional[int],
    machine_operating: Optional[int],
    gps_valid: Optional[int],
) -> List[str]:
    flags: List[str] = []

    if event_time is None:
        flags.append("missing_event_time")
    if not device_id:
        flags.append("missing_device_id")

    if (latitude is None) ^ (longitude is None):
        flags.append("partial_position")

    if latitude is not None and not (-90 <= latitude <= 90):
        flags.append("invalid_latitude_range")

    if longitude is not None and not (-180 <= longitude <= 180):
        flags.append("invalid_longitude_range")

    if machine_operating == 1 and machine_on == 0:
        flags.append("operating_while_off")

    if gps_valid == 1 and (latitude is None or longitude is None):
        flags.append("gps_valid_without_position")

    return flags


# ============================================================
# LEITURA RAW
# ============================================================

def list_jsonl_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Pasta raw não encontrada: {root.resolve()}")
    return sorted(root.rglob("*.jsonl"))


def read_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = str(filepath)
                obj["_source_line"] = line_number
                rows.append(obj)
            except json.JSONDecodeError:
                rows.append({
                    "_source_file": str(filepath),
                    "_source_line": line_number,
                    "_line_parse_error": "invalid_jsonl_line",
                    "_raw_line": line,
                })

    return rows


# ============================================================
# PARSE ASCII DO ZEFIRO
# ============================================================

def parse_ascii_kv_payload(raw_payload: str) -> Dict[str, str]:
    """
    Converte string no formato:
    CHAVE=VALOR|CHAVE=VALOR|...
    em dict.

    Mantém valores vazios.
    """
    result: Dict[str, str] = {}

    if raw_payload is None:
        return result

    parts = str(raw_payload).split("|")
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
        else:
            # parte sem '=' -> ignora
            continue

    return result


def parse_modbus_error(err_value: Optional[str]) -> Dict[str, int]:
    if err_value is None:
        return {
            "gps_lost_flag": 0,
            "empty_modbus_resp_flag": 0,
            "delayed_send_flag": 0,
        }

    err = str(err_value).strip().upper()

    return {
        "gps_lost_flag": 1 if "GNSS_LOST" in err else 0,
        "empty_modbus_resp_flag": 1 if "MB_EMPTY_RESP" in err else 0,
        "delayed_send_flag": 1 if "DELAYED_SEND" in err else 0,
    }


# ============================================================
# PARSER ZEFIRO
# ============================================================

def parse_zefiro_record(envelope: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "normalized_id": make_normalized_id(),
        "hardware": "ZEFIRO",
        "device_id": None,
        "hardware_session_id": None,
        "session_id": None,
        "source_system": None,
        "payload_format": "ascii",

        "state_client": None,
        "city": None,
        "execution_place": None,
        "seed_type": None,
        "weight_kg": None,

        "event_time": None,
        "ingestion_time": None,
        "event_date": None,
        "event_hour": None,
        "delay_ms": None,
        "is_late": None,
        "is_buffered": None,

        "latitude": None,
        "longitude": None,
        "latitude_ref": None,
        "longitude_ref": None,
        "gps_valid": None,
        "gps_status": None,
        "speed_kmh": None,
        "rssi_dbm": None,
        "signal_quality": None,
        "network_condition": None,

        "machine_on": None,
        "machine_operating": None,
        "machine_status": None,
        "scenario_state": None,

        "parse_ok": 0,
        "parse_error": None,
        "data_quality_ok": 0,
        "data_quality_flags": None,
        "raw_record_id": None,
        "raw_payload": None,
        "observacoes": None,

        # auxiliares úteis para debug do Zefiro
        "modem_status": None,
        "gnss_age": None,
        "err_code": None,

        "_source_file": envelope.get("_source_file"),
        "_source_line": envelope.get("_source_line"),
    }

    if envelope.get("_line_parse_error"):
        normalized["parse_error"] = envelope["_line_parse_error"]
        normalized["raw_payload"] = envelope.get("_raw_line")
        normalized["data_quality_flags"] = "invalid_jsonl_line"
        return normalized

    # metadados do envelope
    normalized["source_system"] = envelope.get("source_system")
    normalized["hardware_session_id"] = envelope.get("hardware_session_id")
    normalized["raw_record_id"] = envelope.get("record_id")
    normalized["city"] = envelope.get("city")
    normalized["state_client"] = envelope.get("state_client")
    normalized["execution_place"] = envelope.get("execution_place")
    normalized["seed_type"] = envelope.get("seed_type")
    normalized["weight_kg"] = to_float(envelope.get("weight_kg"))
    normalized["latitude_ref"] = to_float(envelope.get("latitude_ref"))
    normalized["longitude_ref"] = to_float(envelope.get("longitude_ref"))
    normalized["delay_ms"] = to_float(envelope.get("delay_ms"))
    normalized["network_condition"] = envelope.get("network_condition")
    normalized["scenario_state"] = envelope.get("scenario_state")
    normalized["observacoes"] = envelope.get("observacoes")

    ingestion_time = parse_datetime(envelope.get("received_at"))
    normalized["ingestion_time"] = ingestion_time

    raw_payload = envelope.get("raw_payload")
    if raw_payload is None:
        normalized["parse_error"] = "missing_raw_payload"
        normalized["data_quality_flags"] = "missing_raw_payload"
        return normalized

    raw_payload_str = str(raw_payload)
    normalized["raw_payload"] = raw_payload_str

    payload_obj = parse_ascii_kv_payload(raw_payload_str)
    if not payload_obj:
        normalized["parse_error"] = "invalid_ascii_payload"
        normalized["data_quality_flags"] = "invalid_ascii_payload"
        return normalized

    # campos nativos
    device_id = payload_obj.get("DEV")
    event_time = parse_datetime(payload_obj.get("TS"))
    latitude = to_float(payload_obj.get("LAT"))
    longitude = to_float(payload_obj.get("LON"))
    rssi_dbm = to_float(payload_obj.get("RSSI"))
    machine_on = to_int01(payload_obj.get("PWR"))
    machine_operating = to_int01(payload_obj.get("RUN"))
    is_buffered = to_int01(payload_obj.get("BUF"))

    modem_status = payload_obj.get("MODEM")
    gnss_age = payload_obj.get("GNSSAGE")
    err_code = payload_obj.get("ERR")

    # velocidade não está claramente padronizada no ASCII do Zefiro
    # então fica None se não houver campo explícito
    speed_kmh = None

    # gps_valid
    err_flags = parse_modbus_error(err_code)
    if err_flags["gps_lost_flag"] == 1:
        gps_valid = 0
        gps_status = "lost"
    else:
        # se LAT/LON existem, tratamos como válido
        if latitude is not None and longitude is not None:
            gps_valid = 1
            gps_status = "valid"
        else:
            gps_valid = 0
            gps_status = "invalid"

    normalized["device_id"] = device_id
    normalized["event_time"] = event_time
    normalized["latitude"] = latitude
    normalized["longitude"] = longitude
    normalized["gps_valid"] = gps_valid
    normalized["gps_status"] = gps_status
    normalized["speed_kmh"] = speed_kmh
    normalized["rssi_dbm"] = rssi_dbm
    normalized["machine_on"] = machine_on
    normalized["machine_operating"] = machine_operating
    normalized["is_buffered"] = is_buffered
    normalized["modem_status"] = modem_status
    normalized["gnss_age"] = gnss_age
    normalized["err_code"] = err_code

    # session_id
    if normalized["hardware_session_id"]:
        normalized["session_id"] = normalized["hardware_session_id"]
    elif device_id and event_time is not None:
        normalized["session_id"] = f"{device_id}_{event_time.strftime('%Y%m%d')}"
    elif device_id:
        normalized["session_id"] = f"{device_id}_unknown_session"

    if event_time is not None:
        normalized["event_date"] = event_time.date().isoformat()
        normalized["event_hour"] = int(event_time.hour)

    # atraso
    if normalized["delay_ms"] is not None:
        normalized["is_late"] = 1 if normalized["delay_ms"] > 5000 else 0

    # delayed_send também conta como atraso semântico
    if err_flags["delayed_send_flag"] == 1:
        normalized["is_late"] = 1

    # qualidade do sinal
    normalized["signal_quality"] = classify_signal_quality(rssi_dbm)

    # estado da máquina
    normalized["machine_status"] = derive_machine_status(machine_on, machine_operating)

    flags = build_quality_flags(
        event_time=event_time,
        device_id=device_id,
        latitude=latitude,
        longitude=longitude,
        machine_on=machine_on,
        machine_operating=machine_operating,
        gps_valid=gps_valid,
    )

    # acrescenta flags específicas do Zefiro
    if err_flags["gps_lost_flag"] == 1:
        flags.append("gps_lost")
    if err_flags["empty_modbus_resp_flag"] == 1:
        flags.append("empty_modbus_response")
    if err_flags["delayed_send_flag"] == 1:
        flags.append("delayed_send")

    if modem_status and modem_status.upper() == "RETRY":
        flags.append("modem_retry")

    normalized["parse_ok"] = 1
    normalized["parse_error"] = None
    normalized["data_quality_ok"] = 1 if len(flags) == 0 else 0
    normalized["data_quality_flags"] = ";".join(flags) if flags else None

    return normalized


# ============================================================
# EXECUÇÃO
# ============================================================

def main() -> None:
    files = list_jsonl_files(RAW_ROOT)
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .jsonl encontrado em {RAW_ROOT.resolve()}")

    parsed_rows: List[Dict[str, Any]] = []

    print(f"Encontrados {len(files)} arquivo(s) JSONL em: {RAW_ROOT}")

    for file in files:
        print(f"Lendo: {file}")
        rows = read_jsonl(file)
        for row in rows:
            parsed_rows.append(parse_zefiro_record(row))

    df = pd.DataFrame(parsed_rows)

    if "event_time" in df.columns:
        df["event_time_dt"] = pd.to_datetime(df["event_time"], format="ISO8601", errors="coerce")
        df = df.sort_values(["event_time_dt", "device_id"], na_position="last").drop(columns=["event_time_dt"])

    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\nSaídas geradas:")
    print(f"- {OUTPUT_PARQUET}")
    print(f"- {OUTPUT_CSV}")

    print("\nResumo:")
    print(f"Total de registros: {len(df)}")

    if "parse_ok" in df.columns:
        parse_ok_rate = df["parse_ok"].mean() * 100 if len(df) else 0
        print(f"Taxa de parse_ok: {parse_ok_rate:.2f}%")

    if "data_quality_ok" in df.columns:
        dq_ok_rate = df["data_quality_ok"].mean() * 100 if len(df) else 0
        print(f"Taxa de data_quality_ok: {dq_ok_rate:.2f}%")

    print("\nPrincipais parse_error:")
    if "parse_error" in df.columns:
        print(df["parse_error"].fillna("SEM_ERRO").value_counts().head(10).to_string())

    print("\nPrincipais data_quality_flags:")
    if "data_quality_flags" in df.columns:
        print(df["data_quality_flags"].fillna("SEM_FLAG").value_counts().head(15).to_string())


if __name__ == "__main__":
    main()