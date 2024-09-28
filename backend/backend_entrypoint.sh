#!/bin/bash

# If the lancedb directory in the volume is empty, copy the data from the image
if [ ! -d "/app/output/20240827-100424/artifacts/lancedb/entity_description_embeddings.lance" ]; then
    echo "Initializing lancedb data volume..."
    cp -r /app/lancedb_backup/* /app/output/20240827-100424/artifacts/lancedb/
fi

# Execute the original command
exec "$@"
