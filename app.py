from detector.argparser import parser
from detector.detect import Detector, logger    

if __name__ == '__main__':
    print("Welcome to GCAM Anamoly Detector!")

    args = parser.parse_args()

    db_path = args.db_path
    queries = args.queries
    mapping_file = args.mapping_file
    aggregate = args.aggregate
    threshold = args.threshold
    interval = args.interval
    include_normal = args.include_normal
    x_label = args.x_label
    divisor_label = args.divisor_label
    scenario_index = args.scenario_index

    detector = Detector(db_path, queries=queries, x_label=x_label, divisor_label=divisor_label,
                        mapping_file=mapping_file, scenario_index=scenario_index,
                        aggregate=aggregate, threshold=threshold, 
                        interval=interval, include_normal=include_normal)

    query_index = args.query_index
    query_label = args.query_label
    graph = args.graph

    if len(query_label) > 0:
        report = detector.detect(query=detector._get_query_label(query_label), graph=graph)
    elif query_index >= 0:
        report = detector.detect(query=detector._get_query_index(query_index), graph=graph)
    else:
        report = detector.detect_all(max_workers=args.workers, graph=graph)
    if not report:
        logger.info("No anamolies detected for any query.")
    else:
        print()
        detector.print_report(report)




## total emissions by region is good for testing

## caching queries

## several ranges of thereshold

## total emissions for all species: co2, nox, boc, f-gases

## within query, you can remap sectors to categories

## few things by sector for some species