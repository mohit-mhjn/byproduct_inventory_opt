// This file enlists the input files and corresponding columns in each with their description
Note :
delimiter used in input files = "."

//--------------------------------------------------------------------------------------------------------------------------

1. Bird Type:
    A bird type defines the size of the bird available after the Harvest and used for processing meat.

    key: bird_type_id

    bird_type_id: A unique index to identify the bird type across the program
    description: Against the corresponding index how does user define the size
    min_weight: Against the corresponding index min weight of the bird considered
    max_weight: Against the corresponding index max weight of the bird considered
    z_value: % distribution of a perticular bird size in the population >> Not being used

// --------------------------------------------------------------------------------------------------------------------------

2. Birds Available
    The data Inventory of the birds avaialable to the processing unit (supply)

    key: date, bird_type_id

    date: Against given date
    bird_type: Bird type index (that redirects to size of bird)
    available: Inventory of the birds (in total number/count of birds)

// --------------------------------------------------------------------------------------------------------------------------

3. Customer
    Data Related to a customer (Customer Master)

    key: customer_id

    customer_id: Index for a customer (uniquely identified for each customer)
    description: Description of the customer against the corresponding number
    priority: Priority of the customer(1/2) { 1: Important | 2: Un-important }
    serv_agrmnt: (0-1) Service level agreement with corresponding customer  

// --------------------------------------------------------------------------------------------------------------------------

4. Cutting Pattern
    Master for bird processing. A procedure by which the bird is cut into parts/product groups

    key: cp_id

    cp_id: cutting pattern index, a unique id by which cutting pattenr is identified
    cp_line: Cutting Patterns are applied on a conveyer line, the description of which is given here. Used for mapping purpose
          Note: a cutting pattern is associated with a line but converse is not true.

    description: Against the corresponding cutting pattern index how does the customer name the cutting pattern
    section_id: on which individual section/set of sections, the corresponding cutting pattern is applied
    capacity: Kilograms per hour processed by the cp >> Not being used
    ops_cost*: cost of cutting pattern processing (unit not yet defined: per hour or per application of cp or per weight)

// --------------------------------------------------------------------------------------------------------------------------

5. Inventory
    Data for inventory avaialble on hand (opening) at T for planning from T untill the planning horizon length
    The inventory table extracted from the ERP is expected to be in the given format

    key: pgroup_id,bird_type_id,product_type,inv_age

    pgroup_id: The unique identifier for a product_group/part of a bird
    bird_type_id: size of the bird as defined in bird type
    product_type: for a product group and size, it is either fresh or frozen
    inv_age: measured in days, How old is the inventory
    q_on_hand: measured in KG's, This is the Quantity on Hand

// --------------------------------------------------------------------------------------------------------------------------

6. post_processing
    Data for various stages in post-processing of bird to create SKUs

    key: machine_id

    machine_id: Unique identifier for processing equipment
    machine_type: Against which stage/process the equipment is used
    description: Null
    capacity: Caoacity of the equipment to process meat in Kilogram per Hour
    ops_cost*: Cost of processing on this machine (unit yet to be defined: per weight or per time)

// --------------------------------------------------------------------------------------------------------------------------

7. Product Group:
    Definition of product_group. product_group is a part of the bird that obtained by applying cutting pattern one or several sections
    (Note that cutting pattern is applied on individual section or set of sections)

    key: pgroup_id

    pgroup_id: a unique identifier for product_group
    description: description, How a customer calls a product_group
    n_parts: Number of parts per bird
    section_id: The set of sections from which the particular product group is extracted
              (Note that: Product group can be extracted from a uniue set
               eg: A product_group that is extracted from section 1 cannot be yielded by section 2)
// --------------------------------------------------------------------------------------------------------------------------

8. Sales Order:
    The orders placed within the planning horizon, available at the customer sku level
    The open orders table from the ERP is expected to be in the given format
    An order placed can be inform of count of SKUs or total weight of SKUs

    key: order_dt,sku_id,customer_id

    order_dt: Date on which the order is to be satisfied (Assumption: pickup from the warehouse)
    bill_number: Unique identifier of the order (Invocing Purpose) May not be required
    customer_id: unique identifier of the customer
    sku_id: Unique Identifier of the SKU, that relates with the sku master to map with product group, size of bird, fresh/frozen, marination(1/0)
    part_count: Quantity of Order in form of number of units
    part_weight: Quantity of Order in form of number of units

    (Note: It is not necessary to place order in both the UOM's either of the one will suffice the requirement, Conversion factor(avg) is available using the yield data)

// --------------------------------------------------------------------------------------------------------------------------

9. Section:
    The distinct sections of bird for cutting pattern, yield and product group definition
    The sections are exclusive of each other (Do not share any part of the bird)

    key: section_id

    section_id: unique identifier for section (lowest value = 1)
    description: How does user/customer will identify section 

// --------------------------------------------------------------------------------------------------------------------------

10. SKU Master:
    Definition of a meat SKU, It is mapped to various indexes to identify its production process

    key: sku_id

    sku_id: Unique identifier for SKU's
    pgroup_id: Related product group to the SKU
    bird_type_id: Related size of bird to the SKU
    product_type: 1/2 (Fresh/Frozen)
    marination: 1/0 (True, False)
    active: Active/Inactive SKU (0/1)
    description: Null
    selling_price: Selling Price of SKU
    holding_cost: Inventory holding cost of the SKU
                  Only defined for Marination = 0 (Assumption: Inventory for Marinated not held as an Imbalance Inventory)
    shelf_life: Shelf Life against the corresponding SKU Inventory
                  Only defined for Marination = 0 (Assumption: Inventory for Marinated not managed as an Imbalance Inventory)

// --------------------------------------------------------------------------------------------------------------------------

11. update_status.json:

    Structure: {"filename": latest updated on}

    The file checks the latest updated timestamp of the cached files indicated by the key in the same
    Note: Used for chache update history

// --------------------------------------------------------------------------------------------------------------------------

12. Yield:
    The % yield of a particular part of the bird of a defined size by a defined cutting pattern

    key: pgroup_id,bird_type_id,cp_id

    cp_id: Applied Cutting Pattern on the a Set of Sections (one or more together)
    section: Identifying set of sections as aforementioned. This is related to product group
    pgroup_id: Cutting Pattern applied on the section will produce a product group (part)
    n_parts: Number of parts per bird of the corresponding product group.
    bird_type_id: Size of bird used
    yield_p: % yield of the part form the bird

    Note: The yield data is used to make a conversion factor as mentioned in the SKU master point.
          The avg signifies that a sku entity can be produced using multiple cutting patterns
          but the difference in the yield of each is not significant, Hence avg of all is taken.
// --------------------------------------------------------------------------------------------------------------------------

13. flex_range
    The set of size ranges used for flexible size type SKU.

    key: range_id

    range_id: A unique identifier for the flexible size type SKU's. (starts with 100+n, That limits the maximum possible size to 99 in number)
    decription: Description of a size range defined by the customer
    bird_type_id: types of bird included in the type range (delimited by ".")
    
// --------------------------------------------------------------------------------------------------------------------------

14. product_type
    The file has description about product type and its index value

    key: product_type
 
    product_type:  Unique integer identifier given to type of product (1 = Fresh/Chilled and 2 = Frozen)
    decription : String value description for product type like Fresh/Chilled etc.

// --------------------------------------------------------------------------------------------------------------------------

15. marination_type
    This file describe various type of marination in the system

    key: marination

    marination: Unique integer identifier given to type of marination 
    description: String value description for marination type

// --------------------------------------------------------------------------------------------------------------------------
