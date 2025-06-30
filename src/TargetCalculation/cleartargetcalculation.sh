for dir in ./src/TargetCalculation/output/sample_*/; do
    find "$dir" -type f ! -name 'target_values.csv' -delete
done